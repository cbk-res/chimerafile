// serve_forward.cpp -- the llama forwarder.  See serve_forward.h for the
// role split: this file owns the router child process and everything that
// crosses the wire to it.

#include "serve_forward.h"

#include <cpp-httplib/httplib.h>

#include <algorithm>
#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <chrono>
#include <fstream>
#include <memory>
#include <sys/socket.h>
#include <sys/wait.h>
#include <spawn.h>

// The GPU mode the serve process resolved in main(); nullptr means
// auto-detect, which we express by leaving LLAMAFILE_GPU unset so the
// child auto-detects too.
extern "C" const char *chimerafile_gpu_mode(void);

namespace serve {

// ── Loopback HTTP ────────────────────────────────────────────────────────

// One exchange with the router child.  The child speaks plain HTTP on
// loopback, so a raw socket is all it takes: open() sends the request and
// reads the status line plus headers (any body bytes that arrived with
// them land in leftover); the body is then read whole (control calls) or
// relayed chunk by chunk (SSE).
struct UpstreamConn {
    int fd = -1;
    int status = 0;
    std::string content_type = "application/json";
    bool chunked = false;  // Transfer-Encoding: chunked, decoded below
    bool done = false;     // chunked stream fully consumed
    std::string raw;       // undecoded socket bytes (chunked framing)
    std::string leftover;  // decoded body bytes awaiting consumption

    // The streaming relay outlives the handle_chat frame (the provider
    // runs once the response handler returns), so ownership of the fd
    // must transfer, not copy: a copy would be closed twice.
    UpstreamConn() = default;
    UpstreamConn(const UpstreamConn &) = delete;
    UpstreamConn &operator=(const UpstreamConn &) = delete;
    UpstreamConn(UpstreamConn &&o) noexcept {
        fd = o.fd;
        o.fd = -1;
        status = o.status;
        content_type = std::move(o.content_type);
        chunked = o.chunked;
        done = o.done;
        raw = std::move(o.raw);
        leftover = std::move(o.leftover);
    }

    void close() {
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
    }
    // One blocking read of decoded body bytes; false on EOF, timeout, or
    // error. Chunked responses (the router's default framing) are decoded
    // here so callers always see the plain body.
    bool read_some(std::string &out) {
        out.clear();
        if (!leftover.empty()) {
            out = std::move(leftover);
            leftover.clear();
            return true;
        }
        if (!chunked) {
            if (fd < 0)
                return false;
            char buf[16384];
            ssize_t n = ::read(fd, buf, sizeof buf);
            if (n <= 0)
                return false;
            out.assign(buf, (size_t)n);
            return true;
        }
        std::string dec;
        for (;;) {
            if (done)
                return false;
            size_t eol = raw.find("\r\n");
            if (eol == std::string::npos) {
                char buf[8192];
                ssize_t n = fd < 0 ? 0 : ::read(fd, buf, sizeof buf);
                if (n <= 0) {
                    done = true;
                    break;
                }
                raw.append(buf, (size_t)n);
                eol = raw.find("\r\n");
                if (eol == std::string::npos)
                    continue;
            }
            std::string size_line = raw.substr(0, eol);
            raw.erase(0, eol + 2);
            size_t semi = size_line.find(';');  // chunk extensions
            if (semi != std::string::npos)
                size_line.erase(semi);
            size_t len = 0;
            try {
                len = std::stoul(size_line, nullptr, 16);
            } catch (...) {
                done = true;
                break;
            }
            if (len == 0) {
                done = true;  // final chunk; trailers are not used
                break;
            }
            while (raw.size() < len + 2) {
                char buf[16384];
                ssize_t n = fd < 0 ? 0 : ::read(fd, buf, sizeof buf);
                if (n <= 0) {
                    done = true;
                    break;
                }
                raw.append(buf, (size_t)n);
            }
            if (raw.size() < len + 2)
                break;  // EOF mid-chunk
            dec.append(raw, 0, len);
            raw.erase(0, len + 2);
            break;  // hand back one chunk per call
        }
        if (!dec.empty()) {
            out = std::move(dec);
            return true;
        }
        return false;
    }
    std::string read_all() {
        std::string out = std::move(leftover), chunk;
        while (read_some(chunk))
            out += chunk;
        return out;
    }
    ~UpstreamConn() { close(); }
};

static bool loopback_open(const std::string &method, const std::string &path,
                          const std::string &body, int port,
                          int recv_timeout_s, UpstreamConn &c,
                          std::string &err,
                          const std::string &content_type =
                              "application/json") {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        err = "socket: " + std::string(strerror(errno));
        return false;
    }
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (::connect(fd, (sockaddr *)&addr, sizeof addr) < 0) {
        err = "connect 127.0.0.1:" + std::to_string(port) + ": " +
              std::string(strerror(errno));
        ::close(fd);
        return false;
    }
    timeval tv {recv_timeout_s, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);

    std::string req = method + " " + path + " HTTP/1.1\r\n" +
                      "Host: 127.0.0.1:" + std::to_string(port) + "\r\n" +
                      "Connection: close\r\n" +
                      "Content-Type: " + content_type + "\r\n";
    if (!body.empty())
        req += "Content-Length: " + std::to_string(body.size()) + "\r\n";
    req += "\r\n" + body;
    size_t sent = 0;
    while (sent < req.size()) {
        ssize_t n = ::send(fd, req.data() + sent, req.size() - sent,
                           MSG_NOSIGNAL);
        if (n <= 0) {
            err = "send: " + std::string(strerror(errno));
            ::close(fd);
            return false;
        }
        sent += (size_t)n;
    }

    std::string head;
    for (;;) {
        char buf[8192];
        ssize_t n = ::read(fd, buf, sizeof buf);
        if (n <= 0) {
            err = "the router closed the connection";
            ::close(fd);
            return false;
        }
        head.append(buf, (size_t)n);
        if (head.find("\r\n\r\n") != std::string::npos)
            break;
    }
    c.fd = fd;
    size_t eoh = head.find("\r\n\r\n");
    size_t eol = head.find("\r\n");
    std::string status_line = head.substr(0, eol);
    size_t sp = status_line.find(' ');
    c.status = sp == std::string::npos ? 0 : atoi(status_line.c_str() + sp + 1);
    size_t p = eol + 2;
    while (p < head.size()) {
        size_t l = head.find("\r\n", p);
        if (l == std::string::npos || l == p)
            break;
        std::string line = head.substr(p, l - p);
        p = l + 2;
        size_t cp = line.find(':');
        if (cp == std::string::npos)
            continue;
        std::string key = to_lower(line.substr(0, cp));
        size_t s = cp + 1;
        while (s < line.size() && line[s] == ' ')
            s++;
        if (key == "content-type")
            c.content_type = line.substr(s);
        else if (key == "transfer-encoding" &&
                 to_lower(line.substr(s)).find("chunked") != std::string::npos)
            c.chunked = true;
    }
    // bytes that arrived with the header are still framed (chunked) or
    // plain, depending on what the loop just learned
    std::string pre = head.substr(eoh + 4);
    if (c.chunked)
        c.raw = std::move(pre);
    else
        c.leftover = std::move(pre);
    return true;
}

static nlohmann::json try_parse(const std::string &body) {
    try {
        return nlohmann::json::parse(body);
    } catch (...) {
        return nlohmann::json();
    }
}

// True when the serve process resolved a GPU (or left it to auto-detect,
// which on a GPU box means a GPU): model loads must then be serialised
// across the two children so two of them never load at once.  "off" is the
// only mode that is definitely GPU-free.
static bool gpu_serialisation_needed() {
    const char *gpu = chimerafile_gpu_mode();
    return !(gpu != nullptr && std::strcmp(gpu, "off") == 0);
}

// The "model" field of one request body: the top-level "model" of a JSON
// body, or the value of the model part of a multipart form (scanned, since
// the serve side rebuilt the form and hands the forwarder only the bytes).
// Empty when absent.
static std::string extract_model(const std::string &body,
                                 const std::string &content_type) {
    if (to_lower(content_type).find("application/json") != std::string::npos) {
        nlohmann::json j = try_parse(body);
        if (j.is_object() && j.contains("model") && j["model"].is_string())
            return j["model"].get<std::string>();
        return std::string();
    }
    const std::string marker = "name=\"model\"";
    size_t pos = body.find(marker);
    if (pos == std::string::npos)
        return std::string();
    size_t start = body.find("\r\n\r\n", pos);  // end of the part's headers
    if (start == std::string::npos)
        return std::string();
    start += 4;
    size_t end = body.find("\r\n", start);  // the field is a single line
    if (end == std::string::npos)
        return std::string();
    return body.substr(start, end - start);
}

// ── Response shaping ─────────────────────────────────────────────────────

static void send_error(httplib::Response &res, int status, const char *type,
                       const char *code, const std::string &message) {
    nlohmann::json out;
    out["error"]["message"] = message;
    out["error"]["type"] = type;
    out["error"]["code"] = code;
    res.status = status;
    res.set_content(out.dump(-1, ' ', false,
                             nlohmann::json::error_handler_t::replace),
                    "application/json");
}

// The child's error bodies use its own shape (numeric code, its own type
// strings); the serve API keeps the shape clients already see, so non-2xx
// responses are rebuilt here.
static void normalize_error(httplib::Response &res, int status,
                            const std::string &body) {
    std::string message = body;
    const char *type = "server_error", *code = "generation_failed";
    bool shaped = false;
    nlohmann::json j = try_parse(body);
    if (j.is_object() && j.contains("error") && j["error"].is_object()) {
        const nlohmann::json &e = j["error"];
        if (e.contains("message") && e["message"].is_string())
            message = e["message"].get<std::string>();
        std::string etype = e.value("type", std::string());
        if (status == 404) {
            type = "invalid_request_error";
            code = "model_not_found";
        } else if (etype == "exceed_context_size_error") {
            type = "invalid_request_error";
            code = "context_length_exceeded";
        } else if ((status == 400 || status == 500) &&
                   (message.find("failed to load") != std::string::npos ||
                    message.find("is not running") != std::string::npos)) {
            // a spawn that failed: the child came back down, so the model
            // is down, not the client
            type = "server_error";
            code = "model_load_failed";
            status = 503;
            message = "model failed to load";
        } else if (status < 500) {
            type = "invalid_request_error";
            code = "invalid_request";
        }
        shaped = true;
    }
    if (!shaped && body.size() > 500)
        message = "upstream error";
    send_error(res, status, type, code, message);
}

// Rewrite the "model" field of one SSE frame (the child echoes the
// canonical name; clients may have asked by alias).  Non-JSON payloads
// pass through untouched.
static std::string rewrite_sse_frame(const std::string &frame,
                                     const std::string &model_id) {
    std::string out;
    size_t pos = 0;
    while (pos <= frame.size()) {
        size_t eol = frame.find('\n', pos);
        size_t len = eol == std::string::npos ? frame.size() - pos : eol - pos;
        std::string line = frame.substr(pos, len);
        if (line.rfind("data: ", 0) == 0) {
            std::string payload = line.substr(6);
            if (payload != "[DONE]") {
                nlohmann::json j = try_parse(payload);
                if (j.is_object()) {
                    j["model"] = model_id;
                    payload = j.dump(-1, ' ', false,
                                     nlohmann::json::error_handler_t::replace);
                }
            }
            out += "data: " + payload;
        } else {
            out += line;
        }
        if (eol == std::string::npos)
            break;
        out += "\n";
        pos = eol + 1;
    }
    return out;
}

// ── The forwarder ────────────────────────────────────────────────────────

bool Forwarder::write_preset_ini(const Config &cfg, const std::string &path,
                                 std::string &err) {
    std::ofstream f(path);
    if (!f) {
        err = "cannot write preset " + path;
        return false;
    }
    for (const auto &m : cfg.models) {
        if (m.engine != EngineKind::Llama || m.is_alias())
            continue;
        f << "[" << m.id << "]\n";
        f << "model = " << m.file << "\n";
        f << "ctx-size = " << m.ctx << "\n";
        f << "n-gpu-layers = " << m.gpu_layers << "\n";
        // config args are CLI-style pairs ("--temp 0.7"); a flag with no
        // value becomes a bare flag
        for (size_t i = 0; i < m.args.size(); i++) {
            const std::string &a = m.args[i];
            if (a.rfind("--", 0) != 0)
                continue;
            std::string key = a.substr(2);
            std::string value = "true";
            if (i + 1 < m.args.size() && m.args[i + 1].rfind("--", 0) != 0) {
                value = m.args[i + 1];
                i++;
            }
            f << key << " = " << value << "\n";
        }
        std::string aliases;
        for (const auto &o : cfg.models)
            if (o.engine == EngineKind::Llama && o.alias_of == m.id)
                aliases += aliases.empty() ? o.id : "," + o.id;
        if (!aliases.empty())
            f << "alias = " << aliases << "\n";
        f << "\n";
    }
    return true;
}

bool Forwarder::init(const std::string &ape, std::string &err) {
    has_llama = false;
    for (const auto &m : cfg.models)
        if (m.engine == EngineKind::Llama) {
            has_llama = true;
            break;
        }
    if (!has_llama)
        return true;
    ape_path = ape;
    work_dir = "/tmp/chimerafile-serve-" + std::to_string((long)getpid());
    if (mkdir(work_dir.c_str(), 0700) != 0 && errno != EEXIST) {
        err = "cannot create " + work_dir;
        return false;
    }
    ini_path = work_dir + "/models.ini";
    log_path = work_dir + "/router.log";
    for (const auto &m : cfg.models)
        if (m.engine == EngineKind::Llama && !m.is_alias())
            recs[m.id];  // one residency record per physical model
    return write_preset_ini(cfg, ini_path, err);
}

// ── Child spawn, shared by the router and media children ─────────────────

// A free loopback port: bind to port 0, let the kernel pick, release.
static int free_loopback_port(std::string &err) {
    int ls = ::socket(AF_INET, SOCK_STREAM, 0);
    if (ls < 0) {
        err = "socket: " + std::string(strerror(errno));
        return 0;
    }
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    socklen_t alen = sizeof a;
    if (::bind(ls, (sockaddr *)&a, sizeof a) < 0 ||
        ::getsockname(ls, (sockaddr *)&a, &alen) < 0) {
        ::close(ls);
        err = "cannot pick a free port";
        return 0;
    }
    int port = ntohs(a.sin_port);
    ::close(ls);
    return port;
}

// The spawn steps every serve child shares: resolve the APE runtime
// (/proc/self/exe), wire the child fd table (stdin from /dev/null,
// stdout+stderr to `log_path`, everything else CLOEXEC so nothing of ours
// is inherited), and build the environment the APE runtime needs
// (LLAMA_APP_CMD/LLAMA_APP_PATH) plus the resolved GPU mode.  args[0] is
// a placeholder replaced by the resolved runtime path.
static bool spawn_child(std::vector<std::string> &args,
                        const std::string &log_path,
                        const std::string &app_cmd,
                        const std::string &ape_path, pid_t &pid,
                        std::string &err) {
    char exe[4096];
    ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1);
    if (n <= 0) {
        err = "cannot resolve /proc/self/exe";
        return false;
    }
    args[0] = std::string(exe, (size_t)n);

    // Child fd table: stdin from /dev/null, stdout+stderr to the child
    // log, and nothing else of ours (the listen socket must not be
    // inherited).
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_addopen(&fa, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_addopen(&fa, STDOUT_FILENO, log_path.c_str(),
                                    O_WRONLY | O_CREAT | O_APPEND, 0644);
    posix_spawn_file_actions_addopen(&fa, STDERR_FILENO, log_path.c_str(),
                                    O_WRONLY | O_CREAT | O_APPEND, 0644);
    // Keep the child out of our file descriptors: everything above 2 (the
    // listen socket, runtime internals) gets CLOEXEC.  fcntl failures are
    // ignored - an fd may close in the meantime, and a closefile action
    // would turn that into a spurious EBADF from posix_spawn.
    DIR *d = opendir("/proc/self/fd");
    if (d) {
        dirent *de;
        while ((de = readdir(d)) != nullptr) {
            if (!isdigit((unsigned char)de->d_name[0]))
                continue;
            int f = atoi(de->d_name);
            if (f > 2)
                fcntl(f, F_SETFD, FD_CLOEXEC);
        }
        closedir(d);
    }

    // Full environment plus the vars the child (and the router child's
    // model grandchildren) need to run as <runtime> <ape> <cmd> ...
    std::vector<std::string> envv;
    for (char **e = environ; *e; e++)
        envv.push_back(*e);
    auto set_var = [&](const char *name, const char *value) {
        std::string prefix = std::string(name) + "=";
        envv.erase(std::remove_if(envv.begin(), envv.end(),
                                  [&](const std::string &s) {
                                      return s.rfind(prefix, 0) == 0;
                                  }),
                   envv.end());
        envv.push_back(prefix + value);
    };
    set_var("LLAMA_APP_CMD", app_cmd.c_str());
    set_var("LLAMA_APP_PATH", ape_path.c_str());
    // The child must resolve the same GPU mode this process did; an
    // inherited LLAMAFILE_GPU would override our auto-detection.
    const char *gpu = chimerafile_gpu_mode();
    if (gpu)
        set_var("LLAMAFILE_GPU", gpu);
    else {
        std::string prefix = "LLAMAFILE_GPU=";
        envv.erase(std::remove_if(envv.begin(), envv.end(),
                                  [&](const std::string &s) {
                                      return s.rfind(prefix, 0) == 0;
                                  }),
                   envv.end());
    }
    std::vector<char *> envp;
    for (auto &s : envv)
        envp.push_back(&s[0]);
    envp.push_back(nullptr);

    std::vector<char *> argv;
    for (auto &s : args)
        argv.push_back(&s[0]);
    argv.push_back(nullptr);

    posix_spawnattr_t attr;
    posix_spawnattr_init(&attr);
    int rc = posix_spawn(&pid, args[0].c_str(), &fa, &attr, argv.data(),
                         envp.data());
    posix_spawnattr_destroy(&attr);
    posix_spawn_file_actions_destroy(&fa);
    if (rc != 0) {
        err = "posix_spawn: " + std::string(strerror(rc));
        return false;
    }
    return true;
}

bool Forwarder::spawn_router(std::string &err) {
    int port = free_loopback_port(err);
    if (port <= 0)
        return false;

    // The APE runtime is what /proc/self/exe points at; it needs the APE
    // file as its first argument, so the child command is
    // <runtime> <ape> llama --server ...
    std::vector<std::string> args = {
        "/proc/self/exe",  // replaced by the resolved runtime path
        ape_path,
        "llama", "--server",
        "--host", "127.0.0.1",
        "--port", std::to_string(port),
        "--models-max", std::to_string(cfg.max_loaded),
        "--models-preset", ini_path,
    };
    pid_t pid;
    if (!spawn_child(args, log_path, "llama", ape_path, pid, err))
        return false;
    router_pid = (int)pid;
    router_port = port;
    serve_log("router child: pid=%d port=%d log=%s\n", (int)pid, port,
              log_path.c_str());
    return true;
}

bool Forwarder::router_healthy() const {
    if (router_port.load() <= 0)
        return false;
    UpstreamConn c;
    std::string err;
    if (!loopback_open("GET", "/health", "", router_port.load(), 5, c, err))
        return false;
    return c.status == 200;
}

bool Forwarder::wait_healthy(long timeout_s) {
    for (long t = 0; t < timeout_s * 2; t++) {
        if (router_healthy())
            return true;
        if (router_pid.load() > 0 &&
            waitpid((pid_t)router_pid.load(), nullptr, WNOHANG) > 0)
            return false;  // the child is gone
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    return false;
}

bool Forwarder::respawn_router() {
    std::lock_guard<std::mutex> lk(mu);
    if (router_pid.load() > 0) {
        int st = 0;
        bool alive = kill((pid_t)router_pid.load(), 0) == 0;
        waitpid((pid_t)router_pid.load(), &st, WNOHANG);  // reap if dead
        if (alive)
            return router_healthy();  // up but maybe wedged: report as-is
    }
    long now = (long)time(nullptr);
    // Bounded respawns: at most five in any five-minute window; past that
    // the router stays down until the window has passed.
    if (respawns >= 5 && now - last_respawn < 300)
        return false;
    std::string err;
    if (!spawn_router(err)) {
        serve_log("router respawn failed: %s\n", err.c_str());
        return false;
    }
    respawns++;
    last_respawn = now;
    serve_log("router child respawned\n");
    return wait_healthy(60);
}

bool Forwarder::start(std::string &err) {
    if (!has_llama)
        return true;
    std::string serr;
    if (!spawn_router(serr)) {
        err = "router child: " + serr;
        return false;
    }
    if (!wait_healthy(60)) {
        err = "router child did not become healthy (log: " + log_path + ")";
        return false;
    }
    running = true;
    sweeper = std::thread([this] { sweep_loop(); });
    return true;
}

void Forwarder::stop() {
    if (!has_llama)
        return;
    running = false;
    if (sweeper.joinable())
        sweeper.join();
    if (router_pid.load() > 0) {
        kill((pid_t)router_pid.load(), SIGTERM);
        int st = 0;
        for (int i = 0; i < 100; i++) {
            if (waitpid((pid_t)router_pid.load(), &st, WNOHANG) > 0)
                break;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
}

// ── The media forwarder ──────────────────────────────────────────────────

bool MediaForwarder::init(const std::string &ape, const std::string &config,
                          std::string &err) {
    has_media = false;
    for (const auto &m : cfg.models)
        if (m.engine != EngineKind::Llama) {
            has_media = true;
            break;
        }
    if (!has_media)
        return true;
    ape_path = ape;
    config_path = config;
    // The same <pid> work dir the router child uses (created by whoever
    // gets here first; both tolerate EEXIST).
    work_dir = "/tmp/chimerafile-serve-" + std::to_string((long)getpid());
    if (mkdir(work_dir.c_str(), 0700) != 0 && errno != EEXIST) {
        err = "cannot create " + work_dir;
        return false;
    }
    log_path = work_dir + "/media.log";
    return true;
}

bool MediaForwarder::spawn_media(std::string &err) {
    int port = free_loopback_port(err);
    if (port <= 0)
        return false;

    // The child re-parses the same config file; --host/--port override its
    // listen entry, so no generated config is needed.
    std::vector<std::string> args = {
        "/proc/self/exe",  // replaced by the resolved runtime path
        ape_path,
        "media",
        "--host", "127.0.0.1",
        "--port", std::to_string(port),
        "--config", config_path,
    };
    pid_t pid;
    if (!spawn_child(args, log_path, "media", ape_path, pid, err))
        return false;
    media_pid = (int)pid;
    media_port = port;
    serve_log("media child: pid=%d port=%d log=%s\n", (int)pid, port,
              log_path.c_str());
    return true;
}

bool MediaForwarder::media_healthy() const {
    if (media_port.load() <= 0)
        return false;
    UpstreamConn c;
    std::string err;
    if (!loopback_open("GET", "/health", "", media_port.load(), 5, c, err))
        return false;
    return c.status == 200;
}

bool MediaForwarder::wait_healthy(long timeout_s) {
    for (long t = 0; t < timeout_s * 2; t++) {
        if (media_healthy())
            return true;
        if (media_pid.load() > 0 &&
            waitpid((pid_t)media_pid.load(), nullptr, WNOHANG) > 0)
            return false;  // the child is gone
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    return false;
}

bool MediaForwarder::respawn_media() {
    std::lock_guard<std::mutex> lk(mu);
    if (media_pid.load() > 0) {
        int st = 0;
        bool alive = kill((pid_t)media_pid.load(), 0) == 0;
        waitpid((pid_t)media_pid.load(), &st, WNOHANG);  // reap if dead
        if (alive)
            return media_healthy();  // up but maybe wedged: report as-is
    }
    long now = (long)time(nullptr);
    // Bounded respawns: at most five in any five-minute window; past that
    // the child stays down until the window has passed.
    if (respawns >= 5 && now - last_respawn < 300)
        return false;
    std::string err;
    if (!spawn_media(err)) {
        serve_log("media respawn failed: %s\n", err.c_str());
        return false;
    }
    respawns++;
    last_respawn = now;
    serve_log("media child respawned\n");
    return wait_healthy(60);
}

bool MediaForwarder::start(std::string &err) {
    if (!has_media)
        return true;
    std::string serr;
    if (!spawn_media(serr)) {
        err = "media child: " + serr;
        return false;
    }
    if (!wait_healthy(60)) {
        err = "media child did not become healthy (log: " + log_path + ")";
        return false;
    }
    running = true;
    sweeper = std::thread([this] { sweep_loop(); });
    return true;
}

void MediaForwarder::stop() {
    if (!has_media)
        return;
    running = false;
    if (sweeper.joinable())
        sweeper.join();
    if (media_pid.load() > 0) {
        kill((pid_t)media_pid.load(), SIGTERM);
        int st = 0;
        for (int i = 0; i < 100; i++) {
            if (waitpid((pid_t)media_pid.load(), &st, WNOHANG) > 0)
                break;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
}

void MediaForwarder::forward(const std::string &path, const std::string &body,
                             const std::string &content_type,
                             httplib::Response &res) {
    if (!has_media) {
        send_error(res, 502, "server_error", "media_unavailable",
                   "no media child (the config has no media models)");
        return;
    }
    // GPU lease (the mirror of Forwarder::ensure_capacity): the request
    // would load the model if it is not already resident.  If the router
    // (llama) child is loading a model and a GPU is in play, defer instead
    // of loading two GPU models at once.
    {
        std::string model = extract_model(body, content_type);
        if (!model.empty() && model_state(model) != "loaded" &&
            other_child_loading && other_child_loading() &&
            gpu_serialisation_needed()) {
            res.set_header("Retry-After", "5");
            send_error(res, 503, "server_error", "model_loading",
                       "the llama child is loading a model; "
                       "retry in a few seconds");
            return;
        }
    }
    UpstreamConn c;
    std::string err;
    // The child loads the model on first use (on its main thread), so the
    // request can sit for a long time, like a chat completion.
    if (!loopback_open("POST", path, body, media_port.load(), 600, c, err,
                       content_type)) {
        if (!respawn_media() ||
            !loopback_open("POST", path, body, media_port.load(), 600, c, err,
                           content_type)) {
            send_error(res, 502, "server_error", "media_unavailable",
                       "the media child is not available");
            return;
        }
    }
    std::string out = c.read_all();
    res.status = c.status;
    res.set_content(
        out,
        c.content_type.empty() ? "application/json" : c.content_type);
}

nlohmann::json MediaForwarder::status() const {
    nlohmann::json out = nlohmann::json::object();
    if (!has_media || media_port.load() <= 0)
        return out;
    UpstreamConn c;
    std::string err;
    if (!loopback_open("GET", "/v1/status", "", media_port.load(), 10, c, err) ||
        c.status != 200)
        return out;
    nlohmann::json body = try_parse(c.read_all());
    if (!body.is_object())
        return out;
    out["queue"] = body.value("queue", nlohmann::json::object());
    out["resident"] = body.value("resident", nlohmann::json::array());
    return out;
}

void MediaForwarder::sweep_loop() {
    while (running) {
        for (int i = 0; i < 10 && running; i++)
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        if (!running)
            break;
        // The lease gate reads the last known rows, so a failed pull (child
        // down) must not clear them: it would look like everything is
        // unloaded.
        nlohmann::json st = status();
        if (!st.is_object() || !st.contains("resident"))
            continue;
        // A respawned child may run a newer APE than this serve (an upgrade
        // in flight): name the skew once instead of failing on it.
        if (st.contains("version") && st["version"].is_string()) {
            std::string v = st["version"].get<std::string>();
            if (v != CHIMERAFILE_VERSION_STRING && v != warned_version) {
                warned_version = v;
                serve_log("media child runs chimerafile %s, this serve is %s; "
                          "restart serve to align them\n",
                          v.c_str(), CHIMERAFILE_VERSION_STRING);
            }
        }
        nlohmann::json rows = st.value("resident", nlohmann::json::array());
        std::lock_guard<std::mutex> lk(status_mu);
        resident_cache = rows;
    }
}

nlohmann::json MediaForwarder::cached_resident() const {
    std::lock_guard<std::mutex> lk(status_mu);
    return resident_cache;
}

std::string MediaForwarder::model_state(const std::string &id) const {
    auto rows = cached_resident();
    if (rows.is_array())
        for (const auto &r : rows)
            if (r.value("id", std::string()) == id)
                return r.value("state", std::string("unloaded"));
    return "unloaded";
}

bool MediaForwarder::any_loading() const {
    auto rows = cached_resident();
    if (rows.is_array())
        for (const auto &r : rows)
            if (r.value("state", std::string()) == "loading")
                return true;
    return false;
}

bool MediaForwarder::preload(const ModelEntry &e, std::string &err) {
    // GPU lease (mirror of forward): defer an explicit load while the
    // router child is loading a model on the same GPU.
    if (model_state(e.id) != "loaded" && other_child_loading &&
        other_child_loading() && gpu_serialisation_needed()) {
        err = "the llama child is loading a model; retry in a few seconds";
        return false;
    }
    UpstreamConn c;
    std::string oerr;
    if (!loopback_open("POST", "/v1/residency/" + e.id, "",
                       media_port.load(), 600, c, oerr)) {
        if (!respawn_media() ||
            !loopback_open("POST", "/v1/residency/" + e.id, "",
                           media_port.load(), 600, c, oerr)) {
            err = "the media child is not available";
            return false;
        }
    }
    if (c.status == 200)
        return true;
    nlohmann::json body = try_parse(c.read_all());
    err = body.is_object() && body.contains("error")
              ? body["error"].value("message", std::string("preload failed"))
              : std::string("preload failed");
    return false;
}

bool MediaForwarder::unload(const ModelEntry &e, std::string &err) {
    UpstreamConn c;
    std::string oerr;
    if (!loopback_open("POST", "/v1/residency/" + e.id + "/unload", "",
                       media_port.load(), 60, c, oerr)) {
        if (!respawn_media() ||
            !loopback_open("POST", "/v1/residency/" + e.id + "/unload", "",
                           media_port.load(), 60, c, oerr)) {
            err = "the media child is not available";
            return false;
        }
    }
    if (c.status == 200)
        return true;
    nlohmann::json body = try_parse(c.read_all());
    err = body.is_object() && body.contains("error")
              ? body["error"].value("message", std::string("unload failed"))
              : std::string("unload failed");
    return false;
}

// One GET /v1/models from the router, reduced to what the front half
// needs: per model name the mapped state, the raw state, and the child's
// port while running (the router reports it in the status args).
nlohmann::json Forwarder::fetch_models() const {
    nlohmann::json out = nlohmann::json::object();
    UpstreamConn c;
    std::string err;
    if (!loopback_open("GET", "/v1/models", "", router_port.load(), 10, c, err)) {
        serve_log("fetch: open failed: %s\n", err.c_str());
        return out;
    }
    std::string mbody = c.read_all();
    nlohmann::json body = try_parse(mbody);
    if (!body.is_object() || !body.contains("data"))
        return out;
    for (const auto &m : body["data"]) {
        std::string name = m.value("id", std::string());
        if (name.empty())
            continue;
        // while a child is spawning or being killed the router briefly
        // reports the model's status as null; treat that as unloaded and
        // skip the args scan (const operator[] on null json would throw)
        const auto &status = m.value("status", nlohmann::json::object());
        std::string v = status.value("value", std::string());
        nlohmann::json st;
        if (v == "loaded")
            st["state"] = "loaded";
        else if (v == "loading")
            st["state"] = "loading";
        else
            st["state"] = "unloaded";
        st["raw"] = v;
        if (status.is_object() && status.contains("exit_code"))
            st["exit_code"] = status["exit_code"];
        int port = 0;
        if (status.is_object() && status.contains("args")) {
            const auto &args = status["args"];
            if (args.is_array())
                for (size_t i = 0; i + 1 < args.size(); i++)
                    if (args[i] == "--port" && args[i + 1].is_string())
                        port = atoi(args[i + 1].get<std::string>().c_str());
        }
        // the args (and the port in them) persist in the router's status
        // after the child exits, so only report the port while it is loaded
        st["port"] = (st["state"] == "loaded" && port) ? port : 0;
        out[name] = std::move(st);
    }
    return out;
}

nlohmann::json Forwarder::cached_models() const {
    std::lock_guard<std::mutex> lk(models_mu);
    return models_cache;
}

bool Forwarder::any_loading() const {
    auto models = cached_models();
    if (!models.is_object())
        return false;
    for (const auto &item : models.items())
        if (item.value().value("state", std::string()) == "loading")
            return true;
    return false;
}

bool Forwarder::wait_for_state(const std::string &id, const std::string &want,
                               long timeout_s, std::string &err) {
    for (long t = 0; t < timeout_s * 10; t++) {
        if (!running) {
            err = "shutting down";
            return false;
        }
        auto models = cached_models();
        auto it = models.find(id);
        if (it != models.end()) {
            if ((*it).value("state", "") == want)
                return true;
            // a failed spawn comes back "unloaded" with a non-zero exit
            // code; fail fast instead of out waiting the full timeout
            if (want == "loaded" && (*it).value("exit_code", 0) != 0) {
                err = "model failed to load (exit code " +
                      std::to_string((*it)["exit_code"].get<int>()) + ")";
                return false;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    err = "timed out waiting for " + id + " to become " + want;
    return false;
}

// The router evicts on its own LRU when it hits models_max, and its LRU
// does not know our pins -- so when the fleet is full and a model is not
// resident, we pick the victim first (never pinned, never in use) and
// wait for it to actually exit before the request goes over.
bool Forwarder::ensure_capacity(const ModelEntry &e, std::string &err) {
    const ModelEntry *p = cfg.primary_of(e);
    if (!p) {
        err = "alias chain broken for " + e.id;
        return false;
    }
    std::lock_guard<std::mutex> lk(mu);
    auto models = cached_models();
    auto st = models.find(p->id);
    std::string state = st == models.end() ? std::string("unloaded")
                                           : (*st).value("state", std::string("unloaded"));
    if (state != "unloaded")
        return true;  // loaded or loading: the router coalesces waiters
    const Rec &rec = recs[p->id];
    if (rec.in_use.load() > 0)
        return true;  // a request for this model is already driving a load
    // GPU lease: this request would start a load (the model is unloaded and
    // nothing is loading it yet).  If the media child is loading a model and
    // a GPU is in play, defer instead of loading two GPU models at once.  The
    // media side has the mirror gate (MediaForwarder::forward/preload).
    if (other_child_loading && other_child_loading() &&
        gpu_serialisation_needed()) {
        err = "the media child is loading a model; retry in a few seconds";
        return false;
    }
    // Slots in the snapshot: a model counts while it is loaded OR loading
    // (the router owns the slot either way), plus any unloaded model a live
    // request is driving a load for.  Counting "loading" keeps admission
    // correct within the snapshot's staleness window: a model whose child
    // is still coming up must not be treated as free.
    int slots = 0;
    for (const auto &m : cfg.models) {
        if (m.engine != EngineKind::Llama || m.is_alias())
            continue;
        auto it = models.find(m.id);
        std::string s = it == models.end() ? std::string("unloaded")
                                           : (*it).value("state", std::string("unloaded"));
        if (s == "loaded" || s == "loading") {
            slots++;
            continue;
        }
        if (m.id != p->id && recs[m.id].in_use.load() > 0)
            slots++;  // being loaded on another request's behalf
    }
    if (slots < cfg.max_loaded)
        return true;
    long best = -1;
    const ModelEntry *victim = nullptr;
    for (const auto &m : cfg.models) {
        if (m.engine != EngineKind::Llama || m.is_alias() || m.pinned)
            continue;
        auto it = models.find(m.id);
        if (it == models.end() || (*it).value("state", "") != "loaded")
            continue;
        const Rec &r = recs[m.id];
        if (r.in_use.load() > 0 || r.evicting.load())
            continue;
        if (best < 0 || r.last_used.load() < best) {
            best = r.last_used.load();
            victim = &m;
        }
    }
    if (!victim) {
        err = "model limit reached and no model can be evicted";
        return false;
    }
    serve_log("admission: evicting %s to make room for %s\n",
              victim->id.c_str(), e.id.c_str());
    Rec &vr = recs[victim->id];
    vr.evicting = true;
    std::string err2;
    bool unposted = false;
    {
        UpstreamConn c;
        std::string body = "{\"model\": \"" + victim->id + "\"}";
        unposted = loopback_open("POST", "/models/unload", body,
                                 router_port.load(), 10, c, err2);
        if (unposted)
            c.read_all();
    }
    bool ok = false;
    if (unposted)
        ok = wait_for_state(victim->id, "unloaded", 60, err2);
    vr.evicting = false;
    if (!ok) {
        err = "evicting " + victim->id + " failed: " + err2;
        return false;
    }
    return true;
}

void Forwarder::sweep_loop() {
    while (running) {
        for (int i = 0; i < 2 && running; i++)
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        if (!running)
            break;
        auto models = fetch_models();
        {
            std::lock_guard<std::mutex> lk(models_mu);
            models_cache = models;
        }
        long now = (long)time(nullptr);
        for (const auto &e : cfg.models) {
            if (e.engine != EngineKind::Llama || e.is_alias() || e.pinned)
                continue;
            int ttl = e.effective_ttl();
            if (ttl <= 0)
                continue;
            Rec &r = recs[e.id];
            if (r.in_use.load() > 0 || r.evicting.load())
                continue;
            auto it = models.find(e.id);
            if (it == models.end() || (*it).value("state", "") != "loaded")
                continue;
            if (now - r.last_used.load() <= ttl)
                continue;
            serve_log("ttl: unloading %s (idle > %ds)\n", e.id.c_str(), ttl);
            r.evicting = true;
            std::string err;
            {
                UpstreamConn c;
                std::string body = "{\"model\": \"" + e.id + "\"}";
                if (loopback_open("POST", "/models/unload", body,
                                  router_port.load(), 10, c, err))
                    c.read_all();
            }
            // The sweeper cannot wait on cached_models() -- it is the
            // thread that fills it -- so poll the router directly and
            // keep the cache fresh as it goes.
            for (long t = 0; t < 300 && running; t++) {
                auto m = fetch_models();
                {
                    std::lock_guard<std::mutex> lk(models_mu);
                    models_cache = m;
                }
                auto mit = m.find(e.id);
                if (mit != m.end() && (*mit).value("state", "") == "unloaded")
                    break;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            r.evicting = false;
        }
    }
}

void Forwarder::handle_chat(httplib::Response &res, const ModelEntry &e,
                            const nlohmann::ordered_json &body) {
    const ModelEntry *p = cfg.primary_of(e);
    if (!p) {
        send_error(res, 500, "server_error", "bad_config",
                   "alias chain broken for " + e.id);
        return;
    }
    // The child is lenient where the serve API is not, so the shapes the
    // serve used to reject get rejected here, before admission: an invalid
    // request must not evict a resident model to make room for itself.
    if (!body.contains("messages") || !body["messages"].is_array() ||
        body["messages"].empty()) {
        send_error(res, 400, "invalid_request_error", "invalid_request",
                   "'messages' must be a non-empty array");
        return;
    }
    if (body.contains("n") && body["n"].is_number() &&
        body["n"].get<int>() != 1) {
        send_error(res, 400, "invalid_request_error", "invalid_request",
                   "'n' > 1 is not supported");
        return;
    }
    if (body.contains("tools") && !body["tools"].is_array()) {
        send_error(res, 400, "invalid_request_error", "invalid_request",
                   "'tools' must be an array");
        return;
    }
    if (body.contains("tool_choice")) {
        const auto &tc = body["tool_choice"];
        std::string verr;
        if (!tc.is_string() && !tc.is_object())
            verr = "'tool_choice' must be a string or object";
        else if (tc.is_object()) {
            std::string type = tc.value("type", std::string());
            if (type != "none" && type != "function")
                verr = "'tool_choice' object must have type 'none' or 'function'";
        }
        if (!verr.empty()) {
            send_error(res, 400, "invalid_request_error", "invalid_request",
                       verr);
            return;
        }
    }

    std::string err;
    if (!ensure_capacity(*p, err)) {
        res.set_header("Retry-After", "5");
        send_error(res, 503, "server_error", "model_loading", err);
        return;
    }
    Rec &rec = recs[p->id];
    rec.in_use.fetch_add(1);
    rec.last_used = (long)time(nullptr);

    nlohmann::ordered_json fwd = body;
    fwd["model"] = e.id;  // the router resolves aliases; replies echo the
                          // canonical name and get rewritten back below
    const bool stream = body.value("stream", false);
    const std::string model_id = e.id;
    const std::string path = "/v1/chat/completions";

    auto open_upstream = [&] (UpstreamConn &c, std::string &err) -> bool {
        if (loopback_open("POST", path, fwd.dump(), router_port.load(), 600, c, err))
            return true;
        std::string reason = err;
        if (respawn_router() &&
            loopback_open("POST", path, fwd.dump(), router_port.load(), 600, c, err))
            return true;
        err = reason;
        return false;
    };

    if (!stream) {
        UpstreamConn c;
        if (!open_upstream(c, err)) {
            rec.in_use.fetch_sub(1);
            send_error(res, 502, "server_error", "router_unavailable", err);
            return;
        }
        std::string resp = c.read_all();
        if (c.status != 200) {
            rec.in_use.fetch_sub(1);
            normalize_error(res, c.status, resp);
            return;
        }
        nlohmann::json out = try_parse(resp);
        if (!out.is_object()) {
            rec.in_use.fetch_sub(1);
            send_error(res, 502, "server_error", "generation_failed",
                       "the router returned a non-JSON response");
            return;
        }
        out["model"] = model_id;
        if (record_chat)
            record_chat(out.value("usage", nlohmann::json::object()),
                        out.value("timings", nlohmann::json::object()));
        res.set_content(out.dump(-1, ' ', false,
                                 nlohmann::json::error_handler_t::replace),
                        "application/json");
        rec.in_use.fetch_sub(1);
        rec.last_used = (long)time(nullptr);
        return;
    }

    UpstreamConn c;
    if (!open_upstream(c, err)) {
        rec.in_use.fetch_sub(1);
        send_error(res, 502, "server_error", "router_unavailable", err);
        return;
    }
    if (c.status != 200) {
        std::string resp = c.read_all();
        rec.in_use.fetch_sub(1);
        normalize_error(res, c.status, resp);
        return;
    }

    // The provider runs on this request's thread; it relays SSE frames as
    // they arrive and is the only place that may touch `up`.
    auto up = std::make_shared<UpstreamConn>(std::move(c));
    res.set_chunked_content_provider("text/event-stream",
        [up, model_id, &rec](size_t, httplib::DataSink &sink) -> bool {
            bool saw_done = false;
            std::string buf = std::move(up->leftover), chunk;
            for (;;) {
                size_t pos = 0;
                while (true) {
                    size_t sep = buf.find("\n\n", pos);
                    if (sep == std::string::npos)
                        break;
                    std::string frame = buf.substr(pos, sep - pos);
                    buf.erase(0, sep + 2);
                    if (frame.rfind("data: [DONE]", 0) == 0)
                        saw_done = true;
                    std::string out = rewrite_sse_frame(frame, model_id) + "\n\n";
                    if (!sink.write(out.data(), out.size())) {
                        up->close();  // client gone: drop the child's stream
                        rec.in_use.fetch_sub(1);
                        return true;
                    }
                    pos = 0;
                }
                if (!up->read_some(chunk))
                    break;  // EOF or read timeout
                buf += chunk;
            }
            if (!saw_done) {
                // the child went away mid-generation (crash, disconnect
                // handling, read timeout); a bare close makes OpenAI
                // clients report "stream ended without finish_reason", so
                // say why instead
                nlohmann::json ej;
                ej["error"]["message"] = "the generation was interrupted";
                ej["error"]["type"] = "server_error";
                ej["error"]["code"] = "generation_failed";
                std::string frame = "data: " +
                    ej.dump(-1, ' ', false,
                            nlohmann::json::error_handler_t::replace) + "\n\n";
                sink.write(frame.data(), frame.size());
            }
            sink.done();
            rec.in_use.fetch_sub(1);
            rec.last_used = (long)time(nullptr);
            return true;
        });
}

bool Forwarder::preload(const ModelEntry &e, std::string &err) {
    const ModelEntry *p = cfg.primary_of(e);
    if (!p) {
        err = "alias chain broken for " + e.id;
        return false;
    }
    if (!ensure_capacity(*p, err))
        return false;
    auto models = fetch_models();
    auto it = models.find(p->id);
    if (it != models.end() && (*it).value("state", "") == "loaded")
        return true;
    std::string uerr;
    UpstreamConn c;
    std::string req = "{\"model\": \"" + p->id + "\"}";
    if (!loopback_open("POST", "/models/load", req, router_port.load(), 10, c, uerr)) {
        err = uerr;
        return false;
    }
    std::string resp = c.read_all();
    if (c.status != 200) {
        err = "the router refused the load: " + resp.substr(0, 200);
        return false;
    }
    return wait_for_state(p->id, "loaded", 300, err);
}

bool Forwarder::unload(const ModelEntry &e, std::string &err) {
    const ModelEntry *p = cfg.primary_of(e);
    if (!p) {
        err = "alias chain broken for " + e.id;
        return false;
    }
    Rec &rec = recs[p->id];
    if (rec.in_use.load() > 0) {
        err = "model is in use";
        return false;
    }
    auto models = fetch_models();
    auto it = models.find(p->id);
    if (it == models.end() || (*it).value("state", "") != "loaded") {
        err = "model not loaded";
        return false;
    }
    std::string uerr;
    UpstreamConn c;
    std::string req = "{\"model\": \"" + p->id + "\"}";
    if (!loopback_open("POST", "/models/unload", req, router_port.load(), 10, c, uerr)) {
        err = uerr;
        return false;
    }
    c.read_all();
    return wait_for_state(p->id, "unloaded", 60, err);
}

nlohmann::json Forwarder::resident_json() const {
    nlohmann::json arr = nlohmann::json::array();
    auto models = fetch_models();
    for (const auto &e : cfg.models) {
        if (e.engine != EngineKind::Llama)
            continue;
        const ModelEntry *p = cfg.primary_of(e);
        nlohmann::json r;
        r["id"] = e.id;
        r["engine"] = "llama";
        if (!p) {
            r["state"] = "unloaded";
            r["port"] = nullptr;
            r["last_used"] = nullptr;
            r["in_use"] = 0;
            r["ttl_s"] = e.effective_ttl();
            r["pinned"] = e.pinned;
            arr.push_back(std::move(r));
            continue;
        }
        auto it = models.find(p->id);
        std::string st = it == models.end() ? std::string("unloaded")
                                            : (*it).value("state", std::string("unloaded"));
        int port = it == models.end() ? 0 : (*it).value("port", 0);
        const Rec &rec = recs.at(p->id);
        r["state"] = st;
        r["port"] = port ? nlohmann::json(port) : nlohmann::json(nullptr);
        r["last_used"] = rec.last_used.load()
                             ? nlohmann::json(rec.last_used.load())
                             : nlohmann::json(nullptr);
        r["in_use"] = rec.in_use.load();
        r["ttl_s"] = e.effective_ttl();
        r["pinned"] = e.pinned;
        arr.push_back(std::move(r));
    }
    return arr;
}

} // namespace serve
