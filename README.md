const solver = b.dependency("solver", .{ .target = target, .optimize = optimize, });

exe.root_module.addImport("solver", solver.module("solver"));
