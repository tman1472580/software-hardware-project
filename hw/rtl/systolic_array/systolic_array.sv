// N x N weight-stationary systolic array: y = a * W
//   a : 1 x N activation vector (INT8), one vector per cycle
//   W : N x N weight matrix (INT8), loaded one row per cycle
//   y : 1 x N result (INT32)
//   N must be at least 2.
//
// Timing: a vector presented with a_valid at clock edge e0 appears on y with
// y_valid after edge e0 + 2N - 2 (latency 2N - 1 cycles). Vectors can be sent
// back to back, one per cycle.
//
// The input skew (row r delayed r cycles) and output deskew (column c delayed
// N-1-c cycles) are done in here, so the caller sees plain, aligned vectors.
//
// Ports are flat buses: element i is bits [i*W +: W].
module systolic_array #(
    parameter int N   = 4,
    parameter int DW  = 8,
    parameter int ACC = 32
) (
    input  logic                clk,
    input  logic                rst_n,
    // weight load: writes row w_row of W
    input  logic                w_load,
    input  logic [$clog2(N)-1:0] w_row,
    input  logic [N*DW-1:0]     w_data,
    // activations in
    input  logic                a_valid,
    input  logic [N*DW-1:0]     a_data,
    // results out
    output logic                y_valid,
    output logic [N*ACC-1:0]    y_data
);
    // a_h[r][c]: activation entering PE(r,c) from the left
    // p_v[r][c]: partial sum entering PE(r,c) from above
    logic signed [DW-1:0]  a_h [N][N+1];
    logic signed [ACC-1:0] p_v [N+1][N];

    // ---------------- input skew: row r delayed by r cycles ----------------
    genvar r, c;
    generate
        for (r = 0; r < N; r++) begin : g_skew
            if (r == 0) begin : g_nodelay
                assign a_h[0][0] = a_data[0 +: DW];
            end else begin : g_delay
                logic signed [DW-1:0] sr [r];
                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        for (int i = 0; i < r; i++) sr[i] <= '0;
                    end else begin
                        sr[0] <= a_data[r*DW +: DW];
                        for (int i = 1; i < r; i++) sr[i] <= sr[i-1];
                    end
                end
                assign a_h[r][0] = sr[r-1];
            end
        end

        // ---------------- PE grid ----------------
        for (c = 0; c < N; c++) begin : g_top
            assign p_v[0][c] = '0;
        end
        for (r = 0; r < N; r++) begin : g_row
            for (c = 0; c < N; c++) begin : g_col
                pe #(.DW(DW), .ACC(ACC)) u_pe (
                    .clk      (clk),
                    .rst_n    (rst_n),
                    .w_load   (w_load && (w_row == r)),
                    .w_in     (w_data[c*DW +: DW]),
                    .a_in     (a_h[r][c]),
                    .psum_in  (p_v[r][c]),
                    .a_out    (a_h[r][c+1]),
                    .psum_out (p_v[r+1][c])
                );
            end
        end

        // ---------------- output deskew: column c delayed N-1-c ----------------
        for (c = 0; c < N; c++) begin : g_deskew
            if (c == N - 1) begin : g_nodelay
                assign y_data[c*ACC +: ACC] = p_v[N][c];
            end else begin : g_delay
                localparam int D = N - 1 - c;
                logic signed [ACC-1:0] sr [D];
                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        for (int i = 0; i < D; i++) sr[i] <= '0;
                    end else begin
                        sr[0] <= p_v[N][c];
                        for (int i = 1; i < D; i++) sr[i] <= sr[i-1];
                    end
                end
                assign y_data[c*ACC +: ACC] = sr[D-1];
            end
        end
    endgenerate

    // ---------------- valid pipeline: 2N - 1 stages ----------------
    logic [2*N-2:0] vld;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) vld <= '0;
        else        vld <= {vld[2*N-3:0], a_valid};
    end
    assign y_valid = vld[2*N-2];
endmodule
