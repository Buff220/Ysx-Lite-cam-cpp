// ysx_camera.cpp — YsxLite P2P Camera Viewer (C++17 + OpenCV)
// Build: g++ -std=c++17 -O2 ysx_camera.cpp -o ysx_camera $(pkg-config --cflags --libs opencv4)
// Run:   ./ysx_camera [broadcast_ip]   (default: 192.168.1.255)

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

// ─── Byte helpers ─────────────────────────────────────────────────────────────

static inline uint16_t r16be(const uint8_t* p) { return (uint16_t(p[0]) << 8) | p[1]; }
static inline uint16_t r16le(const uint8_t* p) { return (uint16_t(p[1]) << 8) | p[0]; }
static inline uint32_t r32be(const uint8_t* p) {
    return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | p[3];
}
static inline uint64_t r64be(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v = (v << 8) | p[i];
    return v;
}
static inline void w16be(uint8_t* p, uint16_t v) { p[0] = v >> 8; p[1] = v & 0xFF; }
static inline void w32be(uint8_t* p, uint32_t v) {
    p[0] = (v >> 24) & 0xFF; p[1] = (v >> 16) & 0xFF;
    p[2] = (v >>  8) & 0xFF; p[3] = v & 0xFF;
}
static inline void w64be(uint8_t* p, uint64_t v) {
    for (int i = 7; i >= 0; i--) { p[i] = v & 0xFF; v >>= 8; }
}
static inline uint16_t u16_swap(uint16_t x) { return ((x & 0xFF00) >> 8) | ((x & 0x00FF) << 8); }

static std::string readString(const uint8_t* p, int maxlen) {
    std::string s(reinterpret_cast<const char*>(p), maxlen);
    auto z = s.find('\0');
    return z != std::string::npos ? s.substr(0, z) : s;
}

// ─── Protocol constants ───────────────────────────────────────────────────────

enum Commands : uint16_t {
    LanSearch   = 61744,
    P2PAlive    = 61920,
    P2PAliveAck = 61921,
    P2pRdy      = 61762,
    DrwAck      = 61905,
    Drw         = 61904,
    PunchPkt    = 61761,
    HelloAck    = 61697,
    Hello       = 61696,
    P2pReq      = 61728,
    LstReq      = 61799,
    PunchTo     = 61760,
    Close       = 61936,
};

enum ControlCommands : uint16_t {
    ConnectUser    = 0x2010,
    ConnectUserAck = 0x2011,
    StartVideo     = 0x1030,
    StartVideoAck  = 0x1031,
    VideoParamSet  = 0x1830,
    VideoParamSetAck = 0x1831,
    DevStatusAck   = 0x0811,
};

static uint16_t ccDest(uint16_t cmd) {
    if (cmd == ConnectUser) return 0xFF00;
    return 0;
}

// ─── Crypto helpers ───────────────────────────────────────────────────────────

static void XqBytesEnc(uint8_t* buf, int len, int rotate) {
    std::vector<uint8_t> tmp(buf, buf + len);
    for (auto& b : tmp) b = (b & 1) ? b - 1 : b + 1;
    for (int i = 0; i < len - rotate; i++) buf[i] = tmp[i + rotate];
    for (int i = 0; i < rotate; i++) buf[len - rotate + i] = tmp[i];
}

static void XqBytesDec(uint8_t* buf, int len, int rotate) {
    std::vector<uint8_t> tmp(buf, buf + len);
    for (auto& b : tmp) b = (b & 1) ? b - 1 : b + 1;
    for (int i = rotate; i < len; i++) buf[i] = tmp[i - rotate];
    for (int i = 0; i < rotate; i++) buf[i] = tmp[len - rotate + i];
}

// ─── Device info ─────────────────────────────────────────────────────────────

struct DevInfo {
    std::string prefix;   // 4 chars
    std::string suffix;
    uint64_t    serialU64;
    std::string serial;
    std::string devId;
};

// ─── Packet builders ──────────────────────────────────────────────────────────

using Buf = std::vector<uint8_t>;

static Buf makeLanSearch() {
    Buf b(4, 0);
    w16be(b.data(), LanSearch);
    w16be(b.data() + 2, 0);
    return b;
}

static Buf makeP2pRdy(const DevInfo& dev) {
    // 4-byte prefix + 8-byte serialU64 + suffix (null-padded to dev.suffix.size())
    uint8_t hdr[20] = {};
    memcpy(hdr, dev.prefix.c_str(), std::min<int>(4, dev.prefix.size()));
    w64be(hdr + 4, dev.serialU64);
    memcpy(hdr + 12, dev.suffix.c_str(), std::min<int>(8, dev.suffix.size()));

    Buf out(24, 0);
    w16be(out.data(), P2pRdy);
    w16be(out.data() + 2, 20);
    memcpy(out.data() + 4, hdr, 20);
    return out;
}

static Buf makeP2pAlive() {
    Buf b(4, 0);
    w16be(b.data(), P2PAlive);
    w16be(b.data() + 2, 0);
    return b;
}

static Buf makeP2pAliveAck() {
    Buf b(4, 0);
    w16be(b.data(), P2PAliveAck);
    w16be(b.data() + 2, 0);
    return b;
}

static Buf makeDrwAck(const uint8_t* pkt) {
    uint16_t pkt_id   = r16be(pkt + 6);
    uint8_t  m_stream = pkt[5];
    Buf out(12, 0);
    w16be(out.data(), DrwAck);
    w16be(out.data() + 2, 6);
    out[4] = 0xD2;
    out[5] = m_stream;
    w16be(out.data() + 6, 1);
    w16be(out.data() + 8, pkt_id);
    return out;
}

struct Session {
    uint16_t outgoingCommandId = 0;
    uint8_t  ticket[4]         = {0, 0, 0, 0};
    std::string dst_ip;
    uint16_t    dst_port;
    int         sock = -1;

    // frame state
    std::vector<uint8_t> curImage;
    uint16_t rcvSeqId    = 0;
    bool     frame_is_bad = false;

    // callbacks
    std::function<void()>                   onLogin;
    std::function<void(std::vector<uint8_t>)> onFrame;
    std::function<void()>                   onDisconnect;

    std::atomic<int64_t> lastRx{0};
    std::atomic<bool>    started{false};
    std::atomic<bool>    running{true};

    void send(const Buf& b) {
        ::sendto(sock, b.data(), b.size(), 0,
                 reinterpret_cast<const sockaddr*>(&dst_addr), sizeof(dst_addr));
    }
    sockaddr_in dst_addr{};
};

static Buf makeDrw(Session& s, uint16_t command, const uint8_t* data, int dataLen) {
    const int TOKEN_LEN = 4, HDR = 16, START_CMD = 0x110A;

    std::vector<uint8_t> encData;
    int payloadLen = TOKEN_LEN;
    if (data && dataLen > 4) {
        encData.assign(data, data + dataLen);
        XqBytesEnc(encData.data(), dataLen, 4);
        payloadLen += dataLen;
    }

    int pktLen = HDR + payloadLen;
    Buf ret(pktLen, 0);
    w16be(ret.data(), Drw);
    w16be(ret.data() + 2, pktLen - 4);
    ret[4] = 0xD1;
    ret[5] = 0; // channel
    w16be(ret.data() + 6, s.outgoingCommandId);
    w16be(ret.data() + 8, START_CMD);
    w16be(ret.data() + 10, command);
    w16be(ret.data() + 12, u16_swap(payloadLen));
    w16be(ret.data() + 14, ccDest(command));
    memcpy(ret.data() + 16, s.ticket, 4);
    if (!encData.empty()) memcpy(ret.data() + 20, encData.data(), encData.size());
    s.outgoingCommandId++;
    return ret;
}

static Buf sendUsrChk(Session& s) {
    std::vector<uint8_t> buf(160, 0);
    const char* user = "admin";
    const char* pass = "admin";
    memcpy(buf.data(),      user, strlen(user));
    memcpy(buf.data() + 32, pass, strlen(pass));
    return makeDrw(s, ConnectUser, buf.data(), 160);
}

static Buf sendStartVideo(Session& s) {
    return makeDrw(s, StartVideo, nullptr, 0);
}

static Buf sendVideoResolution(Session& s, int resol) {
    std::vector<uint8_t> preset = {1, 0, 0, 0, 2, 0, 0, 0}; // resol 2
    return makeDrw(s, VideoParamSet, preset.data(), preset.size());
}

// ─── Frame reassembly ─────────────────────────────────────────────────────────

static const uint8_t FRAME_HEADER[4] = {0x55, 0xAA, 0x15, 0xA8};
static const uint8_t JPEG_HEADER[4]  = {0xFF, 0xD8, 0xFF, 0xDB};

static void dealWithData(Session& s, const uint8_t* pkt, int /*msgLen*/) {
    uint16_t pkt_len = r16be(pkt + 2);
    if (pkt_len < 12) return;
    uint16_t pkt_id = r16be(pkt + 6);
    const uint8_t* m_hdr = pkt + 8;

    auto startNewFrame = [&](const uint8_t* data, int len) {
        if (!s.curImage.empty() && !s.frame_is_bad) {
            if (s.onFrame) s.onFrame(s.curImage);
        }
        s.frame_is_bad = false;
        s.curImage.assign(data, data + len);
        s.rcvSeqId = pkt_id;
    };

    if (memcmp(m_hdr, FRAME_HEADER, 4) == 0) {
        uint8_t stream_type = pkt[12];
        if (stream_type == 3) { // JPEG
            int to_read = pkt_len - 4 - 32;
            if (to_read > 0) startNewFrame(pkt + 40, to_read);
        }
        // skip audio (stream_type 6)
    } else {
        int data_len = pkt_len - 4;
        const uint8_t* data = pkt + 8;
        if (memcmp(data, JPEG_HEADER, 4) == 0) {
            startNewFrame(data, data_len);
        } else {
            if (pkt_id <= s.rcvSeqId) return;
            if (pkt_id > s.rcvSeqId + 1) { s.frame_is_bad = true; return; }
            s.rcvSeqId = pkt_id;
            if (!s.curImage.empty())
                s.curImage.insert(s.curImage.end(), data, data + data_len);
        }
    }
}

// ─── Control command handler ─────────────────────────────────────────────────

static void handleControlCmd(Session& s, const uint8_t* pkt, int msgLen) {
    uint16_t cmd_id     = r16be(pkt + 10);
    int      payload_len = r16le(pkt + 12);
    if (msgLen > 20 && payload_len > msgLen) payload_len = msgLen - 20;
    if (payload_len > 4) {
        // decrypt in-place — copy since pkt may be const
        std::vector<uint8_t> tmp(pkt, pkt + msgLen);
        XqBytesDec(tmp.data() + 20, payload_len - 4, 4);
        if (cmd_id == ConnectUserAck) {
            memcpy(s.ticket, tmp.data() + 24, 4);
            if (s.onLogin) s.onLogin();
        }
    } else if (cmd_id == ConnectUserAck && !s.onLogin) {
        // fallback: ticket stays zero
        if (s.onLogin) s.onLogin();
    }
}

// ─── Incoming packet router ───────────────────────────────────────────────────

static void handleIncoming(Session& s, const uint8_t* msg, int len) {
    if (len < 4) return;
    uint16_t raw = r16be(msg);
    s.lastRx.store(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());

    switch (raw) {
    case P2pRdy:
        s.send(sendUsrChk(s));
        break;
    case P2PAlive:
        s.send(makeP2pAlive());
        break;
    case P2PAliveAck:
        break;
    case Drw: {
        s.send(makeDrwAck(msg));
        uint8_t m_stream = msg[5];
        if (m_stream == 1) {
            dealWithData(s, msg, len);
        } else if (m_stream == 0) {
            handleControlCmd(s, msg, len);
        }
        break;
    }
    case DrwAck: {
        // nothing to do for unacked map in this simple impl
        break;
    }
    case Close:
        s.send(makeDrwAck(msg));
        if (s.onDisconnect) s.onDisconnect();
        break;
    default:
        break;
    }
}

// ─── Session lifecycle ────────────────────────────────────────────────────────

static std::shared_ptr<Session> makeSession(const DevInfo& dev,
                                             const std::string& ip, uint16_t port) {
    auto s = std::make_shared<Session>();
    s->dst_ip   = ip;
    s->dst_port = port;

    s->dst_addr.sin_family = AF_INET;
    s->dst_addr.sin_port   = htons(port);
    inet_pton(AF_INET, ip.c_str(), &s->dst_addr.sin_addr);

    s->sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (s->sock < 0) throw std::runtime_error("socket() failed");

    sockaddr_in bind_addr{};
    bind_addr.sin_family      = AF_INET;
    bind_addr.sin_addr.s_addr = INADDR_ANY;
    bind_addr.sin_port        = 0;
    bind(s->sock, reinterpret_cast<sockaddr*>(&bind_addr), sizeof(bind_addr));

    // receive thread
    std::thread([s, dev]() mutable {
        uint8_t buf[65536];
        sockaddr_in from{};
        socklen_t fromlen = sizeof(from);

        s->send(makeP2pRdy(dev));
        s->started.store(true);

        while (s->running.load()) {
            fd_set fds; FD_ZERO(&fds); FD_SET(s->sock, &fds);
            timeval tv{0, 100000}; // 100 ms
            if (select(s->sock + 1, &fds, nullptr, nullptr, &tv) > 0) {
                int n = recvfrom(s->sock, buf, sizeof(buf), 0,
                                 reinterpret_cast<sockaddr*>(&from), &fromlen);
                if (n > 0) handleIncoming(*s, buf, n);
            }
        }
    }).detach();

    // keepalive / timeout thread
    std::thread([s]() {
        while (s->running.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(400));
            if (!s->started.load()) continue;
            auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            int64_t delta = now - s->lastRx.load();
            if (delta > 600) s->send(makeP2pAlive());
            if (delta > 5000) {
                std::cout << "[" << "cam" << "] Timeout, disconnecting\n";
                if (s->onDisconnect) s->onDisconnect();
                break;
            }
        }
    }).detach();

    s->onLogin = [s, dev]() {
        std::cout << "[" << dev.devId << "] Logged in, starting video...\n";
        s->send(sendVideoResolution(*s, 2));
        s->send(sendStartVideo(*s));
    };

    return s;
}

// ─── Discovery ───────────────────────────────────────────────────────────────

struct DiscoveredCamera {
    sockaddr_in rinfo{};
    DevInfo     dev;
};

using DiscoverCb = std::function<void(DiscoveredCamera)>;

static void discoverDevices(const std::string& broadcast_ip, DiscoverCb cb) {
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) throw std::runtime_error("discovery socket() failed");

    int yes = 1;
    setsockopt(sock, SOL_SOCKET, SO_BROADCAST, &yes, sizeof(yes));
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    sockaddr_in bind_addr{};
    bind_addr.sin_family      = AF_INET;
    bind_addr.sin_addr.s_addr = INADDR_ANY;
    bind_addr.sin_port        = 0;
    bind(sock, reinterpret_cast<sockaddr*>(&bind_addr), sizeof(bind_addr));

    sockaddr_in dst{};
    dst.sin_family = AF_INET;
    dst.sin_port   = htons(32108);
    inet_pton(AF_INET, broadcast_ip.c_str(), &dst.sin_addr);

    auto ls = makeLanSearch();

    auto sendSearch = [&]() {
        sendto(sock, ls.data(), ls.size(), 0,
               reinterpret_cast<sockaddr*>(&dst), sizeof(dst));
    };

    std::cout << "Searching for cameras on " << broadcast_ip << "...\n";
    sendSearch();

    // periodic re-broadcast
    std::thread([sock, sendSearch]() mutable {
        while (true) {
            std::this_thread::sleep_for(std::chrono::seconds(3));
            sendSearch();
        }
    }).detach();

    // receive loop
    std::thread([sock, cb]() {
        uint8_t buf[65536];
        sockaddr_in from{};
        socklen_t fromlen = sizeof(from);

        while (true) {
            fd_set fds; FD_ZERO(&fds); FD_SET(sock, &fds);
            timeval tv{0, 500000};
            if (select(sock + 1, &fds, nullptr, nullptr, &tv) <= 0) continue;

            int n = recvfrom(sock, buf, sizeof(buf), 0,
                             reinterpret_cast<sockaddr*>(&from), &fromlen);
            if (n < 4) continue;

            if (r16be(buf) == PunchPkt) {
                // parse PunchPkt
                uint16_t len = r16be(buf + 2);
                if (n < 20) continue;
                DevInfo dev;
                dev.prefix    = readString(buf + 4, 4);
                dev.serialU64 = r64be(buf + 8);
                dev.serial    = std::to_string(dev.serialU64);
                int suffLen   = std::max(0, (int)len - 16 + 4);
                dev.suffix    = suffLen > 0 ? readString(buf + 16, std::min(suffLen, n - 16)) : "";
                dev.devId     = dev.prefix + dev.serial + dev.suffix;

                DiscoveredCamera cam;
                cam.rinfo = from;
                cam.dev   = dev;
                cb(cam);
            }
        }
    }).detach();
}

// ─── Main: show frames via OpenCV ────────────────────────────────────────────

int main(int argc, char* argv[]) {
    std::string broadcast_ip = (argc > 1) ? argv[1] : "192.168.1.255";

    // Map of devId → session
    std::map<std::string, std::shared_ptr<Session>> sessions;
    std::mutex sessions_mtx;

    // OpenCV display: we share a frame buffer per window name
    struct FrameState {
        std::vector<uint8_t> jpeg;
        std::mutex           mtx;
        bool                 fresh = false;
    };
    std::map<std::string, std::shared_ptr<FrameState>> frames;
    std::mutex frames_mtx;

    discoverDevices(broadcast_ip, [&](DiscoveredCamera cam) {
        std::lock_guard<std::mutex> lock(sessions_mtx);
        if (sessions.count(cam.dev.devId)) return;

        char ipbuf[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &cam.rinfo.sin_addr, ipbuf, sizeof(ipbuf));
        uint16_t port = ntohs(cam.rinfo.sin_port);
        std::cout << "Discovered camera " << cam.dev.devId << " at " << ipbuf << "\n";

        // Create per-camera frame state
        auto fs = std::make_shared<FrameState>();
        {
            std::lock_guard<std::mutex> fl(frames_mtx);
            frames[cam.dev.devId] = fs;
        }

        auto s = makeSession(cam.dev, ipbuf, port);

        s->onFrame = [fs](std::vector<uint8_t> jpeg) {
            std::lock_guard<std::mutex> l(fs->mtx);
            fs->jpeg  = std::move(jpeg);
            fs->fresh = true;
        };

        s->onDisconnect = [&sessions, &sessions_mtx, devId = cam.dev.devId]() {
            std::lock_guard<std::mutex> lock(sessions_mtx);
            sessions.erase(devId);
        };

        sessions[cam.dev.devId] = s;
    });

    // Main thread: OpenCV display loop (must run on main thread on macOS/Linux)
    std::cout << "Press ESC or Q in any window to quit.\n";
    while (true) {
        {
            std::lock_guard<std::mutex> fl(frames_mtx);
            for (auto& [devId, fs] : frames) {
                std::lock_guard<std::mutex> l(fs->mtx);
                if (!fs->fresh || fs->jpeg.empty()) continue;
                fs->fresh = false;

                // Decode JPEG
                std::vector<uint8_t> jpegCopy = fs->jpeg;
                cv::Mat img = cv::imdecode(jpegCopy, cv::IMREAD_COLOR);
                if (img.empty()) continue;

                cv::imshow(devId, img);
            }
        }

        int key = cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') break;

        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    cv::destroyAllWindows();
    return 0;
}