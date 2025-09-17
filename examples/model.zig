const std = @import("std");
const math = std.math;
const ModelInterface = @import("nmpc_solver.zig").ModelInterface;

pub const UnicycleModel = struct {
    const Self = @This();
    dt: f64,
    Q: [3]f64,
    R: [2]f64,
    x_ref: [3]f64,

    pub fn init(dt: f64, Q: [3]f64, R: [2]f64, x_ref: [3]f64) Self {
        return .{
            .dt = dt,
            .Q = Q,
            .R = R,
            .x_ref = x_ref,
        };
    }

    // Implementasi model dinamika unicycle robot
    pub fn dynamics(self: *const Self, state: []const f64, control: []const f64, result: []f64) void {
        // State: [x_pos, y_pos, theta]
        const x_pos = state[0];
        const y_pos = state[1];
        const theta = state[2];
        // Control: [v (linear velocity), omega (angular velocity)]
        const v = control[0];
        const omega = control[1];
        // Model dinamika diskret (Forward Euler)
        result[0] = x_pos + self.dt * v * math.cos(theta); // x_next
        result[1] = y_pos + self.dt * v * math.sin(theta); // y_next
        result[2] = theta + self.dt * omega; // theta_next
    }

    // Implementasi stage cost function
    pub fn stage_cost(self: *const Self, state: []const f64, control: []const f64) f64 {
        // State error cost: (x - x_ref)^T * Q * (x - x_ref)
        var state_cost: f64 = 0.0;
        for (0..3) |i| {
            const errors = state[i] - self.x_ref[i];
            state_cost += self.Q[i] * errors * errors;
        }
        // Control cost: u^T * R * u
        var control_cost: f64 = 0.0;
        for (0..2) |i| {
            control_cost += self.R[i] * control[i] * control[i];
        }
        return state_cost + control_cost;
    }

    // Wrapper functions untuk interface
    fn dynamics_wrapper(model: *const anyopaque, state: []const f64, control: []const f64, result: []f64) void {
        const self: *const UnicycleModel = @ptrCast(@alignCast(model));
        self.dynamics(state, control, result);
    }

    fn stage_cost_wrapper(model: *const anyopaque, state: []const f64, control: []const f64) f64 {
        const self: *const UnicycleModel = @ptrCast(@alignCast(model));
        return self.stage_cost(state, control);
    }

    // Helper function untuk membuat ModelInterface
    pub fn toInterface(self: *const Self) ModelInterface {
        return .{
            .dynamics_fn = dynamics_wrapper,
            .stage_cost_fn = stage_cost_wrapper,
            .model_ptr = @ptrCast(self),
        };
    }
};
