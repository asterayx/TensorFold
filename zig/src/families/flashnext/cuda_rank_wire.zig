//! Bounded Linux TCP control channel. Tensor bytes travel over NCCL; this channel owns command order and peer EOF.
const std = @import("std");
const builtin = @import("builtin");
const platform = @import("sys");
const p = std.posix;
const net = std.Io.net;

pub const max_payload = 1 << 20;
/// Ranks of one group at most: the leader and its followers.
pub const max_ranks = 8;
pub const Kind = enum(u32) { hello = 1, ready, create, forward, round, drop_seq, save, restore, drop_snap, stop, ack, verify, keep, media, verify_many };
pub const Header = struct {
    kind: Kind,
    sequence: u64 = 0,
    object: u64 = 0,
    arg: u64 = 0,
    bytes: u32 = 0,
    flags: u32 = 0,

    pub fn encode(h: Header) ![40]u8 {
        if (h.bytes > max_payload) return error.BadRankFrame;
        var out: [40]u8 = @splat(0);
        @memcpy(out[0..4], "TFR1");
        std.mem.writeInt(u32, out[4..8], @backingInt(h.kind), .little);
        std.mem.writeInt(u64, out[8..16], h.sequence, .little);
        std.mem.writeInt(u64, out[16..24], h.object, .little);
        std.mem.writeInt(u64, out[24..32], h.arg, .little);
        std.mem.writeInt(u32, out[32..36], h.bytes, .little);
        std.mem.writeInt(u32, out[36..40], h.flags, .little);
        return out;
    }

    pub fn decode(raw: *const [40]u8) !Header {
        if (!std.mem.eql(u8, raw[0..4], "TFR1")) return error.BadRankFrame;
        const h: Header = .{
            .kind = std.enums.fromInt(Kind, std.mem.readInt(u32, raw[4..8], .little)) orelse return error.BadRankFrame,
            .sequence = std.mem.readInt(u64, raw[8..16], .little),
            .object = std.mem.readInt(u64, raw[16..24], .little),
            .arg = std.mem.readInt(u64, raw[24..32], .little),
            .bytes = std.mem.readInt(u32, raw[32..36], .little),
            .flags = std.mem.readInt(u32, raw[36..40], .little),
        };
        if (h.bytes > max_payload) return error.BadRankFrame;
        return h;
    }
};

pub fn now() u64 {
    var ts: platform.c.timespec = undefined;
    if (platform.c.clock_gettime(platform.c.CLOCK.MONOTONIC, &ts) != 0 or ts.sec < 0) return 0;
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_ms * 1000 + @as(u64, @intCast(ts.nsec));
}

pub fn deadline(ms: u64) u64 {
    return now() +| (ms *| std.time.ns_per_ms);
}

pub fn pause() void {
    const ts: platform.c.timespec = .{ .sec = 0, .nsec = 50_000 };
    _ = platform.c.nanosleep(&ts, null);
}

fn wait(fd: p.socket_t, events: i16, until: ?u64) !void {
    while (true) {
        const current = now();
        if (until) |end| if (current >= end) return error.RankTimeout;
        const ms: i32 = if (until) |end| @intCast(@min(1000, (end - current + std.time.ns_per_ms - 1) / std.time.ns_per_ms)) else 1000;
        var fds = [_]p.pollfd{.{ .fd = fd, .events = events, .revents = 0 }};
        const count = try p.poll(&fds, ms);
        if (count == 0) continue;
        if (fds[0].revents & events != 0) return;
        if (fds[0].revents & (p.POLL.ERR | p.POLL.HUP | p.POLL.NVAL) != 0) return error.RankDisconnected;
    }
}

/// A follower's link to the leader, or the leader's links to every follower: a star, rank 0 at its center.
pub const Link = struct {
    /// The leader (on a follower), or the first follower (on the leader).
    fd: p.socket_t,
    /// The leader's other followers; a follower has none.
    more: [max_ranks - 2]p.socket_t = @splat(-1),
    extra: u8 = 0,

    /// The processes at the other end: one on a follower, every follower on the leader.
    pub fn peers(l: Link) usize {
        return 1 + @as(usize, l.extra);
    }

    /// The link to peer `i` alone.
    pub fn peer(l: Link, i: usize) Link {
        return .{ .fd = if (i == 0) l.fd else l.more[i - 1] };
    }

    /// The leader's links in the order `order` names them (`order[i]` is the peer that becomes peer i).
    pub fn ordered(l: Link, order: []const usize) Link {
        var out: Link = .{ .fd = l.peer(order[0]).fd, .extra = l.extra };
        for (order[1..], 0..) |from, i| out.more[i] = l.peer(from).fd;
        return out;
    }

    /// Ends both directions of every link without closing it: a blocked receive on either side returns.
    pub fn shutdown(l: Link) void {
        for (0..l.peers()) |i| _ = std.c.shutdown(l.peer(i).fd, p.SHUT.RDWR);
    }

    pub fn close(l: Link) void {
        l.shutdown();
        for (0..l.peers()) |i| _ = std.c.close(l.peer(i).fd);
    }

    /// `world` rank processes: rank 0 accepts the others in the order they connect (the hello names their ranks).
    /// Retry only connection establishment, never an accepted command.
    pub fn open(io: std.Io, host: []const u8, port: u16, rank: usize, world: usize, timeout_ms: u64) !Link {
        if (builtin.os.tag != .linux) return error.RankPlatform;
        if (world < 2 or world > max_ranks or rank >= world or port == 0) return error.BadRank;
        const address = net.IpAddress.parse(host, port) catch try net.IpAddress.resolve(io, host, port);
        var storage: std.Io.Threaded.PosixAddress = undefined;
        const len = std.Io.Threaded.addressToPosix(&address, &storage);
        const family: u32 = if (address == .ip4) p.AF.INET else p.AF.INET6;
        const until = deadline(timeout_ms);
        if (rank == 0) {
            const fd = try socket(family);
            defer _ = std.c.close(fd);
            const one: c_int = 1;
            try p.setsockopt(fd, p.SOL.SOCKET, p.SO.REUSEADDR, std.mem.asBytes(&one));
            if (std.c.bind(fd, &storage.any, len) != 0) return error.RankBind;
            if (std.c.listen(fd, @intCast(world - 1)) != 0) return error.RankListen;
            var got: [max_ranks - 1]p.socket_t = undefined;
            var n: usize = 0;
            errdefer for (got[0..n]) |x| {
                _ = std.c.close(x);
            };
            while (n < world - 1) {
                try wait(fd, p.POLL.IN, until);
                const accepted = std.c.accept4(fd, null, null, p.SOCK.NONBLOCK | p.SOCK.CLOEXEC);
                switch (p.errno(accepted)) {
                    .SUCCESS => {
                        got[n] = accepted;
                        n += 1;
                        try noDelay(accepted);
                    },
                    .INTR, .AGAIN => continue,
                    else => return error.RankAccept,
                }
            }
            var out: Link = .{ .fd = got[0], .extra = @intCast(n - 1) };
            @memcpy(out.more[0 .. n - 1], got[1..n]);
            return out;
        }
        while (now() < until) {
            const fd = try socket(family);
            const rc = std.c.connect(fd, &storage.any, len);
            if (rc == 0) return .{ .fd = fd };
            const e = p.errno(rc);
            if (e == .INPROGRESS or e == .INTR) {
                wait(fd, p.POLL.OUT, until) catch |err| {
                    _ = std.c.close(fd);
                    if (err == error.RankTimeout) return err;
                    pause();
                    continue;
                };
                var status: c_int = 0;
                var size: p.socklen_t = @sizeOf(c_int);
                if (std.c.getsockopt(fd, p.SOL.SOCKET, p.SO.ERROR, &status, &size) == 0 and status == 0) return .{ .fd = fd };
            }
            _ = std.c.close(fd);
            pause();
        }
        return error.RankTimeout;
    }

    /// The same bytes to every peer: the leader's commands reach each follower in the same order.
    pub fn send(l: Link, bytes: []const u8, until: u64) !void {
        for (0..l.peers()) |i| try sendOne(l.peer(i).fd, bytes, until);
    }

    fn sendOne(fd: p.socket_t, bytes: []const u8, until: u64) !void {
        if (builtin.os.tag == .windows) return error.RankPlatform; // no MSG.NOSIGNAL or DONTWAIT in Winsock
        var at: usize = 0;
        while (at < bytes.len) {
            if (now() >= until) return error.RankTimeout;
            const rc = std.c.send(fd, bytes.ptr + at, bytes.len - at, p.MSG.NOSIGNAL);
            switch (p.errno(rc)) {
                .SUCCESS => {
                    if (rc == 0) return error.RankDisconnected;
                    at += @intCast(rc);
                },
                .INTR => continue,
                .AGAIN => try wait(fd, p.POLL.OUT, until),
                else => return error.RankDisconnected,
            }
        }
    }

    /// From the one peer of this link; the leader of several followers reads each through `peer`.
    pub fn receive(l: Link, bytes: []u8, until: ?u64) !void {
        if (builtin.os.tag == .windows) return error.RankPlatform;
        if (l.extra != 0) return error.BadRank;
        var at: usize = 0;
        while (at < bytes.len) {
            if (until) |end| if (now() >= end) return error.RankTimeout;
            const rc = std.c.recv(l.fd, bytes.ptr + at, bytes.len - at, 0);
            switch (p.errno(rc)) {
                .SUCCESS => {
                    if (rc == 0) return error.RankDisconnected;
                    at += @intCast(rc);
                },
                .INTR => continue,
                .AGAIN => try wait(l.fd, p.POLL.IN, until),
                else => return error.RankDisconnected,
            }
        }
    }

    pub fn sendHeader(l: Link, h: Header, until: u64) !void {
        const raw = try h.encode();
        try l.send(&raw, until);
    }

    pub fn receiveHeader(l: Link, until: ?u64) !Header {
        var raw: [40]u8 = undefined;
        // An idle follower can wait indefinitely; a partial frame cannot.
        try l.receive(raw[0..1], until);
        try l.receive(raw[1..], until orelse deadline(30_000));
        return Header.decode(&raw);
    }

    /// A non-consuming EOF check while a GPU collective progresses. A future control frame stays queued.
    pub fn checkPeer(l: Link) !void {
        if (builtin.os.tag == .windows) return error.RankPlatform;
        for (0..l.peers()) |i| {
            var byte: u8 = 0;
            const rc = std.c.recv(l.peer(i).fd, &byte, 1, p.MSG.PEEK | p.MSG.DONTWAIT);
            if (rc == 0) return error.RankDisconnected;
            if (rc < 0) switch (p.errno(rc)) {
                .AGAIN, .INTR => {},
                else => return error.RankDisconnected,
            };
        }
    }
};

fn socket(family: u32) !p.socket_t {
    const fd = std.c.socket(family, p.SOCK.STREAM | p.SOCK.NONBLOCK | p.SOCK.CLOEXEC, 0);
    if (p.errno(fd) != .SUCCESS) return error.RankSocket;
    errdefer _ = std.c.close(fd);
    try noDelay(fd);
    return fd;
}

fn noDelay(fd: p.socket_t) !void {
    const one: c_int = 1;
    try p.setsockopt(fd, p.IPPROTO.TCP, p.TCP.NODELAY, std.mem.asBytes(&one));
}

test "rank frames are versioned, bounded and exact" {
    const h: Header = .{ .kind = .forward, .sequence = 700, .object = 19, .arg = 42, .bytes = max_payload, .flags = 3 };
    var raw = try h.encode();
    try std.testing.expectEqualDeep(h, try Header.decode(&raw));
    raw[0] = 0;
    try std.testing.expectError(error.BadRankFrame, Header.decode(&raw));
    raw = try h.encode();
    @memset(raw[4..8], 0xff);
    try std.testing.expectError(error.BadRankFrame, Header.decode(&raw));
    raw = try h.encode();
    @memset(raw[32..36], 0xff);
    try std.testing.expectError(error.BadRankFrame, Header.decode(&raw));
    try std.testing.expectError(error.BadRankFrame, (Header{ .kind = .forward, .bytes = max_payload + 1 }).encode());
}

test "a leader's link sends to every follower, reads each alone and notices any of them leaving" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    var pairs: [3][2]c_int = undefined;
    for (&pairs) |*pair| try std.testing.expectEqual(@as(c_int, 0), std.c.socketpair(p.AF.UNIX, p.SOCK.STREAM | p.SOCK.NONBLOCK, 0, pair));
    var leader: Link = .{ .fd = pairs[0][0], .extra = 2 };
    leader.more[0] = pairs[1][0];
    leader.more[1] = pairs[2][0];
    defer leader.close();
    var open_ends = [_]bool{ true, true, true };
    defer for (pairs, open_ends) |pair, still| {
        if (still) _ = std.c.close(pair[1]);
    };
    try std.testing.expectEqual(@as(usize, 3), leader.peers());
    const until = deadline(2_000); // nonblocking ends: a frame that never comes is a failed test, not a hang
    try leader.sendHeader(.{ .kind = .stop, .sequence = 7 }, until);
    for (pairs) |pair| {
        const follower: Link = .{ .fd = pair[1] };
        try std.testing.expectEqual(@as(usize, 1), follower.peers());
        try std.testing.expectEqual(@as(u64, 7), (try follower.receiveHeader(until)).sequence);
    }
    // a follower's answer is read from its own link, and the whole link refuses a read that names no follower
    try (Link{ .fd = pairs[2][1] }).sendHeader(.{ .kind = .ack, .sequence = 9 }, until);
    try std.testing.expectEqual(@as(u64, 9), (try leader.peer(2).receiveHeader(until)).sequence);
    var byte: [1]u8 = undefined;
    try std.testing.expectError(error.BadRank, leader.receive(&byte, until));
    // rank order from connection order
    const again = leader.ordered(&.{ 2, 0, 1 });
    try std.testing.expectEqual(pairs[2][0], again.peer(0).fd);
    try std.testing.expectEqual(pairs[0][0], again.peer(1).fd);
    try std.testing.expectEqual(pairs[1][0], again.peer(2).fd);
    try leader.checkPeer();
    _ = std.c.close(pairs[1][1]);
    open_ends[1] = false;
    try std.testing.expectError(error.RankDisconnected, leader.checkPeer());
}
