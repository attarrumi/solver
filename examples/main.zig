const std = @import("std");
const nmpc = @import("nmpc_solver.zig");
const math = std.math;

const UnicycleModel = @import("model.zig").UnicycleModel;

const SimulationData = struct {
    step: usize,
    x: f64,
    y: f64,
    theta: f64,
    v: f64,
    omega: f64,
    cost: f64,
    target_x: f64,
    target_y: f64,
    target_theta: f64,
    computation_time_ns: u64,
    fgm_iterations: usize,
    bfgs_iterations: usize,
    converged: bool,
};

// Fungsi untuk menghitung jumlah iterasi FGM dan BFGS dari trace
fn countIterations(trace: []nmpc.Method) struct { fgm: usize, bfgs: usize } {
    var fgm: usize = 0;
    var bfgs: usize = 0;

    for (trace) |method| {
        switch (method) {
            .FGM => fgm += 1,
            .BFGS => bfgs += 1,
        }
    }

    return .{ .fgm = fgm, .bfgs = bfgs };
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Parameter sistem
    const nx: usize = 3; // jumlah state [x, y, theta]
    const nu: usize = 2; // jumlah control input [v, omega]
    const N: usize = 15; // horizon prediksi

    var unicycle_model = UnicycleModel.init(
        0.1,
        [_]f64{ 50.0, 50.0, 0.1 },
        [_]f64{ 0.01, 0.01 },
        [_]f64{ 5.0, 3.0, 0.0 },
    );

    // Konversi ke ModelInterface
    const model_interface = unicycle_model.toInterface();

    // Inisialisasi solver NMPC
    var solver = try nmpc.NMPC_Solver.init(allocator, nx, nu, N, model_interface);
    defer solver.deinit();

    // Parameter solver dari GA
    solver.params.lr = 0.001;
    solver.params.fgm_steps = 100;
    solver.params.bfgs_steps = 50;
    solver.params.tol = 1e-2;
    solver.params.max_line_search = 10;
    solver.params.bfgs_memory = 50;

    // State awal robot
    var current_state = [_]f64{ 0.0, 0.0, 0.0 }; // posisi awal di origin

    std.log.info("=== NMPC Unicycle Robot Control ===", .{});
    std.log.info("Initial state: x={d:.2}, y={d:.2}, theta={d:.2}", .{ current_state[0], current_state[1], current_state[2] });
    std.log.info("Target state: x={d:.2}, y={d:.2}, theta={d:.2}", .{ unicycle_model.x_ref[0], unicycle_model.x_ref[1], unicycle_model.x_ref[2] });

    // Simulasi kontrol loop
    var step: usize = 0;
    const max_steps: usize = 70;
    var simulation_data = try std.ArrayList(SimulationData).initCapacity(allocator, max_steps);
    defer simulation_data.deinit(allocator);

    while (step < max_steps) : (step += 1) {
        // Hitung kontrol optimal menggunakan NMPC
        var timer = try std.time.Timer.start();
        var result = try solver.optimize(&current_state);
        defer result.deinit(allocator);

        const computation_time = timer.lap();

        // Hitung jumlah iterasi FGM dan BFGS
        const iterations = countIterations(result.method_trace);

        // Ambil kontrol input pertama dari sekuens optimal (MPC principle)
        const optimal_control = [_]f64{ result.U_opt[0], result.U_opt[1] };

        // Simpan data simulasi
        try simulation_data.append(allocator, SimulationData{
            .step = step,
            .x = current_state[0],
            .y = current_state[1],
            .theta = current_state[2],
            .v = optimal_control[0],
            .omega = optimal_control[1],
            .cost = result.final_cost,
            .target_x = unicycle_model.x_ref[0],
            .target_y = unicycle_model.x_ref[1],
            .target_theta = unicycle_model.x_ref[2],
            .computation_time_ns = computation_time,
            .fgm_iterations = iterations.fgm,
            .bfgs_iterations = iterations.bfgs,
            .converged = result.converged,
        });

        // std.log.info("Step {d}: State=[{d:.3}, {d:.3}, {d:.3}] Control=[{d:.3}, {d:.3}] Cost={d:.6}", .{ step, current_state[0], current_state[1], current_state[2], optimal_control[0], optimal_control[1], result.final_cost });

        // Simulasi sistem dengan kontrol yang dihitung
        var next_state: [3]f64 = undefined;
        solver.model.dynamics(&current_state, &optimal_control, &next_state);
        current_state = next_state;

        // Cek konvergensi
        const position_error = math.sqrt(math.pow(f64, current_state[0] - unicycle_model.x_ref[0], 2) +
            math.pow(f64, current_state[1] - unicycle_model.x_ref[1], 2));
        const angle_error = @abs(current_state[2] - unicycle_model.x_ref[2]);

        if (position_error < 0.01 and angle_error < 0.05) {
            // std.log.info("Target reached after {d} steps!", .{step + 1});
            // break;
        }

        // Update referensi jika diperlukan (contoh: tracking trajectory)
        if (step > 20) {
            unicycle_model.x_ref[0] = 8.0; // ubah target x
            unicycle_model.x_ref[1] = 5.0; // ubah target y
            // std.log.info("Target changed to: x={d:.2}, y={d:.2}", .{ model.x_ref[0], model.x_ref[1] });
        }

        if (step > 50) {
            unicycle_model.x_ref[0] = 3.0; // ubah target x
            unicycle_model.x_ref[1] = 9.0; // ubah target y
            // std.log.info("Target changed to: x={d:.2}, y={d:.2}", .{ model.x_ref[0], model.x_ref[1] });
        }
    }

    // std.log.info("Final state: x={d:.3}, y={d:.3}, theta={d:.3}", .{ current_state[0], current_state[1], current_state[2] });

    // Simpan data ke file CSV
    const filename = "simulation_results.csv";
    var file = try std.fs.cwd().createFile(filename, .{});
    defer file.close();

    try file.writeAll("step,x,y,theta,v,omega,cost,target_x,target_y,target_theta,computation_time_ms,fgm_iterations,bfgs_iterations,converged\n");

    // Tulis data
    for (simulation_data.items) |data| {
        const line = try std.fmt.allocPrint(allocator, "{d},{d:.6},{d:.6},{d:.6},{d:.6},{d:.6},{d:.6},{d:.6},{d:.6},{d:.6},{d:.3},{d},{d},{d}\n", .{
            data.step,
            data.x,
            data.y,
            data.theta,
            data.v,
            data.omega,
            data.cost,
            data.target_x,
            data.target_y,
            data.target_theta,
            @as(f64, @floatFromInt(data.computation_time_ns)) / 1_000_000.0, // Convert to milliseconds
            data.fgm_iterations,
            data.bfgs_iterations,
            if (data.converged) @as(u32, 1) else @as(u32, 0),
        });
        defer allocator.free(line);
        try file.writeAll(line);
    }

    std.log.info("Data simulasi disimpan ke {s}", .{filename});
}
