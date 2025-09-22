const std = @import("std");
const math = std.math;
const linalg = @import("linalg.zig");

pub const ModelInterface = struct {
    const Self = @This();

    // Function pointers untuk interface
    dynamics_fn: *const fn (model: *const anyopaque, state: []const f32, control: []const f32, result: []f32) void,
    stage_cost_fn: *const fn (model: *const anyopaque, state: []const f32, control: []const f32) f32,

    // Pointer ke model instance
    model_ptr: *const anyopaque,

    pub fn dynamics(self: Self, state: []const f32, control: []const f32, result: []f32) void {
        self.dynamics_fn(self.model_ptr, state, control, result);
    }

    pub fn stage_cost(self: Self, state: []const f32, control: []const f32) f32 {
        return self.stage_cost_fn(self.model_ptr, state, control);
    }
};

pub const Method = enum {
    FGM,
    BFGS,
};

/// Hasil utama dari optimize()
pub const OptimizeResult = struct {
    U_opt: []f32,
    method_trace: []Method,
    iterations: usize,
    final_cost: f32,
    converged: bool,

    pub fn deinit(self: *OptimizeResult, allocator: std.mem.Allocator) void {
        allocator.free(self.U_opt);
        allocator.free(self.method_trace);
    }
};

pub const NMPC_Solver = struct {
    const Self = @This();

    // Runtime parameters
    nx: usize,
    nu: usize,
    N: usize,

    allocator: std.mem.Allocator,

    last_U_opt: ?[]f32 = null,
    // Parameter solver
    params: struct {
        U_min: f32 = -12.0,
        U_max: f32 = 12.0,
        lr: f32 = 0.01,
        fgm_steps: usize = 35,
        bfgs_steps: usize = 25,
        tol: f32 = 1e-4,
        max_line_search: usize = 50,
        bfgs_memory: usize = 10,
    } = .{},

    model: ModelInterface,

    pub fn init(
        allocator: std.mem.Allocator,
        nx: usize,
        nu: usize,
        N: usize,
        model: ModelInterface,
    ) !Self {
        return Self{
            .allocator = allocator,
            .nx = nx,
            .nu = nu,
            .N = N,
            .model = model,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.last_U_opt) |last_u| {
            self.allocator.free(last_u);
            self.last_U_opt = null;
        }
    }

    // ---------------- METODE SOLVER ----------------

    fn rollout(self: Self, x0: []const f32, U: []const f32, X: [][]f32) void {
        @memcpy(X[0], x0);
        var k: usize = 0;
        while (k < self.N) : (k += 1) {
            const u_slice = U[k * self.nu .. (k + 1) * self.nu];
            self.model.dynamics(X[k], u_slice, X[k + 1]);
        }
    }

    fn objectiveNMPC(self: Self, U: []const f32, x0: []const f32, workspace: [][]f32) f32 {
        self.rollout(x0, U, workspace);
        var cost: f32 = 0.0;
        var k: usize = 0;
        while (k < self.N) : (k += 1) {
            const u_slice = U[k * self.nu .. (k + 1) * self.nu];
            cost += self.model.stage_cost(workspace[k], u_slice);

            for (u_slice) |u_val| {
                if (u_val < self.params.U_min) {
                    cost += 100 * math.pow(f32, self.params.U_min - u_val, 2);
                } else if (u_val > self.params.U_max) {
                    cost += 100 * math.pow(f32, u_val - self.params.U_max, 2);
                }
            }
        }
        return cost;
    }

    // IMPROVEMENT: Function signature changed to return an error to handle allocation failures.
    fn gradient(self: Self, U: []const f32, x0: []const f32, grad: []f32, workspace: [][]f32) !void {
        const eps: f32 = 1e-6;

        const U_perturbed = try self.allocator.alloc(f32, U.len);
        defer self.allocator.free(U_perturbed);

        @memcpy(U_perturbed, U);

        for (0..U.len) |i| {
            const old_val = U[i];

            U_perturbed[i] = old_val + eps;
            const f_plus = self.objectiveNMPC(U_perturbed, x0, workspace);

            U_perturbed[i] = old_val - eps;
            const f_minus = self.objectiveNMPC(U_perturbed, x0, workspace);

            grad[i] = (f_plus - f_minus) / (2.0 * eps);

            U_perturbed[i] = old_val; // restore
        }
    }

    // FIX: Entire lbfgsOptimize function is heavily refactored for correctness and efficiency.
    fn lbfgsOptimize(
        self: *Self,
        x0: []const f32,
        U: []f32, // Takes ownership and modifies in place
        trace: *std.ArrayList(Method),
        workspace: [][]f32,
        iter_count: *usize,
    ) !void {
        const n = U.len;

        // --- IMPROVEMENT: Allocate L-BFGS history buffers once ---
        var s_list = try self.allocator.alloc([]f32, self.params.bfgs_memory);
        defer self.allocator.free(s_list);
        for (0..self.params.bfgs_memory) |i| s_list[i] = try self.allocator.alloc(f32, n);
        defer for (s_list) |s| self.allocator.free(s);

        var y_list = try self.allocator.alloc([]f32, self.params.bfgs_memory);
        defer self.allocator.free(y_list);
        for (0..self.params.bfgs_memory) |i| y_list[i] = try self.allocator.alloc(f32, n);
        defer for (y_list) |y| self.allocator.free(y);

        var rho_list = try self.allocator.alloc(f32, self.params.bfgs_memory);
        defer self.allocator.free(rho_list);

        // --- IMPROVEMENT: Allocate workspace vectors once before the loop ---
        const g = try self.allocator.alloc(f32, n);
        defer self.allocator.free(g);
        const g_old = try self.allocator.alloc(f32, n);
        defer self.allocator.free(g_old);
        const U_old = try self.allocator.alloc(f32, n);
        defer self.allocator.free(U_old);
        const q = try self.allocator.alloc(f32, n);
        defer self.allocator.free(q);
        const p = try self.allocator.alloc(f32, n);
        defer self.allocator.free(p);
        var alpha_hist = try self.allocator.alloc(f32, self.params.bfgs_memory);
        defer self.allocator.free(alpha_hist);

        try self.gradient(U, x0, g, workspace);

        var hist_count: usize = 0;
        var i: usize = 0;
        while (i < self.params.bfgs_steps) : (i += 1) {
            iter_count.* += 1;
            const grad_norm = linalg.norm(g);
            if (grad_norm < self.params.tol) break;

            // --- L-BFGS two-loop recursion ---
            @memcpy(q, g);

            // Backward loop
            if (hist_count > 0) {
                var j: isize = @as(isize, @intCast(hist_count)) - 1;
                while (j >= 0) : (j -= 1) {
                    const idx: usize = @intCast(j);
                    alpha_hist[idx] = rho_list[idx] * linalg.dot(s_list[idx], q);
                    linalg.axpy(-alpha_hist[idx], y_list[idx], q); // q = q - alpha * y
                }
            }

            // Scaling initial Hessian approximation
            if (hist_count > 0) {
                const last_idx = hist_count - 1;
                const gamma = linalg.dot(s_list[last_idx], y_list[last_idx]) / linalg.dot(y_list[last_idx], y_list[last_idx]);
                linalg.scale(gamma, q); // q = gamma * q
            }

            // Forward loop
            if (hist_count > 0) {
                var j: usize = 0;
                while (j < hist_count) : (j += 1) {
                    const beta = rho_list[j] * linalg.dot(y_list[j], q);
                    linalg.axpy(alpha_hist[j] - beta, s_list[j], q); // q = q + (alpha - beta) * s
                }
            }

            // Search direction p = -Hk * g  (where q is Hk * g)
            linalg.scaleTo(-1.0, q, p);

            // --- Backtracking Line Search ---
            const current_cost = self.objectiveNMPC(U, x0, workspace);
            var step_size: f32 = 1.0;
            const c1: f32 = 1e-4;
            const m = linalg.dot(g, p); // Directional derivative

            const U_trial = try self.allocator.alloc(f32, n); // Temporary for line search
            defer self.allocator.free(U_trial);

            var ls_success = false;
            for (0..self.params.max_line_search) |_| {
                @memcpy(U_trial, U);
                linalg.axpy(step_size, p, U_trial); // U_trial = U + step_size * p
                const f_new = self.objectiveNMPC(U_trial, x0, workspace);

                if (f_new <= current_cost + c1 * step_size * m) {
                    ls_success = true;
                    break;
                }
                step_size *= 0.5;
            }

            if (!ls_success) {
                // std.log.debug("Line search failed.", .{});
                break;
            }

            // --- Update state for next iteration ---
            @memcpy(U_old, U);
            @memcpy(g_old, g);
            @memcpy(U, U_trial);

            try self.gradient(U, x0, g, workspace);

            // --- Update L-BFGS history (s and y vectors) ---
            const current_idx = if (hist_count < self.params.bfgs_memory) hist_count else hist_count % self.params.bfgs_memory;

            // FIX: Correct calculation for s_k = U_k - U_{k-1}
            linalg.sub(U, U_old, s_list[current_idx]);
            // FIX: Correct calculation for y_k = g_k - g_{k-1}
            linalg.sub(g, g_old, y_list[current_idx]);

            const s_dot_y = linalg.dot(s_list[current_idx], y_list[current_idx]);

            if (s_dot_y > 1e-9) { // Ensure curvature condition
                // FIX: Calculate rho only once here
                rho_list[current_idx] = 1.0 / s_dot_y;
                if (hist_count < self.params.bfgs_memory) {
                    hist_count += 1;
                } else {
                    // This logic is for a circular buffer, which is simpler
                    // than shifting all elements. We just overwrite the oldest entry.
                    // To implement shifting, you'd need to rotate the slices.
                    // For simplicity, we use modulo arithmetic on current_idx.
                }
            }

            try trace.append(self.allocator, .BFGS);
        }
    }

    fn adaptive_parameters(self: *Self, x: []const f32) void {
        const pos_error = @abs(x[0]);
        const angle_error = if (self.nx > 2) @abs(x[2]) else 0.0;

        // Adjust parameters based on current state
        if (pos_error > 1.0 or angle_error > 0.1) {
            // Aggressive mode - large errors need more aggressive control
            self.params.lr = 0.15;
            self.params.fgm_steps = 40;
            self.params.bfgs_steps = 15;
            self.params.tol = 1e-4;
        } else if (pos_error > 0.1 or angle_error > 0.05) {
            // Medium mode - moderate errors
            self.params.lr = 0.08;
            self.params.fgm_steps = 30;
            self.params.bfgs_steps = 10;
            self.params.tol = 1e-5;
        } else {
            // Fine-tuning mode - small errors, focus on precision
            self.params.lr = 0.03;
            self.params.fgm_steps = 20;
            self.params.bfgs_steps = 5;
            self.params.tol = 1e-6;
        }

        // std.log.debug("Adaptive params: lr={d}, fgm_steps={d}, bfgs_steps={d}", .{
        //     self.params.lr,
        //     self.params.fgm_steps,
        //     self.params.bfgs_steps,
        // });
    }

    pub fn optimize(
        self: *Self,
        x0: []const f32,
    ) !OptimizeResult {
        // self.adaptive_parameters(x0);

        const workspace = try self.allocator.alloc([]f32, self.N + 1);
        defer self.allocator.free(workspace);
        for (workspace) |*state| state.* = try self.allocator.alloc(f32, self.nx);
        defer for (workspace) |state| self.allocator.free(state);

        const U = if (self.last_U_opt) |last_U| blk: {
            const shifted_U = try self.allocator.alloc(f32, self.N * self.nu);
            const nu = self.nu;
            const N = self.N;
            // Shift previous solution
            @memcpy(shifted_U[0 .. (N - 1) * nu], last_U[nu .. N * nu]);
            // Extrapolate last control input
            @memcpy(shifted_U[(N - 1) * nu .. N * nu], shifted_U[(N - 2) * nu .. (N - 1) * nu]);
            break :blk shifted_U;
        } else blk: {
            break :blk try self.allocator.alloc(f32, self.N * self.nu);
        };
        // U will be owned by OptimizeResult, so we only free it on error paths
        var err_free_U = true;
        defer if (err_free_U) self.allocator.free(U);

        var trace = std.ArrayList(Method).initCapacity(self.allocator, self.params.fgm_steps + self.params.bfgs_steps) catch unreachable;
        defer trace.deinit(self.allocator);

        var total_iterations: usize = 0;
        var converged = false;
        var final_cost: f32 = 0.0;

        const g = try self.allocator.alloc(f32, U.len);
        defer self.allocator.free(g);

        // ---------- FGM ----------
        var fgm_iter: usize = 0;
        while (fgm_iter < self.params.fgm_steps) : (fgm_iter += 1) {
            try self.gradient(U, x0, g, workspace);
            const norm_g = linalg.norm(g);
            final_cost = self.objectiveNMPC(U, x0, workspace);

            if (norm_g < self.params.tol) {
                converged = true;
                break;
            }

            linalg.axpy(-self.params.lr, g, U); // U = U - lr * g
            linalg.clampVec(U, self.params.U_min, self.params.U_max);
            try trace.append(self.allocator, .FGM);
            total_iterations += 1;
        }

        // ---------- L-BFGS ----------
        if (!converged) {
            // FIX: Pass total_iterations pointer to be updated by lbfgsOptimize
            try self.lbfgsOptimize(x0, U, &trace, workspace, &total_iterations);

            final_cost = self.objectiveNMPC(U, x0, workspace);
            try self.gradient(U, x0, g, workspace);
            converged = linalg.norm(g) < self.params.tol;
        }

        // Update last optimal solution
        if (self.last_U_opt) |old_u| self.allocator.free(old_u);
        self.last_U_opt = try self.allocator.dupe(f32, U);

        // Success, so transfer ownership of U to OptimizeResult
        err_free_U = false;
        return OptimizeResult{
            .U_opt = U,
            .method_trace = try trace.toOwnedSlice(self.allocator),
            .iterations = total_iterations,
            .final_cost = final_cost,
            .converged = converged,
        };
    }
};
