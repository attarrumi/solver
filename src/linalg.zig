const std = @import("std");
const math = std.math;

pub fn norm(vec: []const f32) f32 {
    var sum: f32 = 0.0;
    for (vec) |v| {
        sum += v * v;
    }
    return @sqrt(sum);
}

pub fn dot(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    var sum: f32 = 0.0;
    for (a, b) |a_val, b_val| {
        sum += a_val * b_val;
    }
    return sum;
}

pub fn clampVec(vec: []f32, min_val: f32, max_val: f32) void {
    for (vec) |*v| {
        v.* = math.clamp(v.*, min_val, max_val);
    }
}

pub fn copy(dest: []f32, src: []const f32) void {
    std.debug.assert(dest.len == src.len);
    @memcpy(dest, src);
}

// Menghitung: vec = scalar * vec (operasi in-place, 2 argumen)
pub fn scale(scalar: f32, vec: []f32) void {
    for (vec) |*v| {
        v.* *= scalar;
    }
}

// Menghitung: dest = scalar * src (operasi out-of-place, 3 argumen)
pub fn scaleTo(scalar: f32, src: []const f32, dest: []f32) void {
    std.debug.assert(src.len == dest.len);
    for (dest, src) |*d_elem, s_elem| {
        d_elem.* = scalar * s_elem;
    }
}

// Menghitung: dest = a + b
pub fn add(a: []const f32, b: []const f32, dest: []f32) void {
    std.debug.assert(dest.len == a.len and a.len == b.len);
    for (dest, a, b) |*d, a_val, b_val| {
        d.* = a_val + b_val;
    }
}

// Menghitung: dest = a - b
pub fn sub(a: []const f32, b: []const f32, dest: []f32) void {
    std.debug.assert(dest.len == a.len and a.len == b.len);
    for (dest, a, b) |*d, a_val, b_val| {
        d.* = a_val - b_val;
    }
}

// Menghitung y = a*x + y
pub fn axpy(a: f32, x: []const f32, y: []f32) void {
    std.debug.assert(x.len == y.len);
    for (y, x) |*y_elem, x_elem| {
        y_elem.* += a * x_elem;
    }
}
