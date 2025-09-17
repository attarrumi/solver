const std = @import("std");
const math = std.math;

pub fn norm(vec: []const f64) f64 {
    var sum: f64 = 0.0;
    for (vec) |v| {
        sum += v * v;
    }
    return @sqrt(sum);
}

pub fn dot(a: []const f64, b: []const f64) f64 {
    std.debug.assert(a.len == b.len);
    var sum: f64 = 0.0;
    for (a, b) |a_val, b_val| {
        sum += a_val * b_val;
    }
    return sum;
}

pub fn clampVec(vec: []f64, min_val: f64, max_val: f64) void {
    for (vec) |*v| {
        v.* = math.clamp(v.*, min_val, max_val);
    }
}

pub fn copy(dest: []f64, src: []const f64) void {
    std.debug.assert(dest.len == src.len);
    @memcpy(dest, src);
}

// Menghitung: vec = scalar * vec (operasi in-place, 2 argumen)
pub fn scale(scalar: f64, vec: []f64) void {
    for (vec) |*v| {
        v.* *= scalar;
    }
}

// Menghitung: dest = scalar * src (operasi out-of-place, 3 argumen)
pub fn scaleTo(scalar: f64, src: []const f64, dest: []f64) void {
    std.debug.assert(src.len == dest.len);
    for (dest, src) |*d_elem, s_elem| {
        d_elem.* = scalar * s_elem;
    }
}

// Menghitung: dest = a + b
pub fn add(a: []const f64, b: []const f64, dest: []f64) void {
    std.debug.assert(dest.len == a.len and a.len == b.len);
    for (dest, a, b) |*d, a_val, b_val| {
        d.* = a_val + b_val;
    }
}

// Menghitung: dest = a - b
pub fn sub(a: []const f64, b: []const f64, dest: []f64) void {
    std.debug.assert(dest.len == a.len and a.len == b.len);
    for (dest, a, b) |*d, a_val, b_val| {
        d.* = a_val - b_val;
    }
}

// Menghitung y = a*x + y
pub fn axpy(a: f64, x: []const f64, y: []f64) void {
    std.debug.assert(x.len == y.len);
    for (y, x) |*y_elem, x_elem| {
        y_elem.* += a * x_elem;
    }
}
