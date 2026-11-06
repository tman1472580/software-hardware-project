// Processing element for a weight-stationary systolic array.
// Holds one INT8 weight. Each cycle:
//   - passes the activation to the right
//   - adds activation * weight to the partial sum coming from above
//     and passes it down
module pe #(
    parameter int DW  = 8,   // data width (activations and weights)
    parameter int ACC = 32   // accumulator width
) (
    input  logic                  clk,
    input  logic                  rst_n,
    // weight load
    input  logic                  w_load,
    input  logic signed [DW-1:0]  w_in,
    // dataflow
    input  logic signed [DW-1:0]  a_in,     // from the left
    input  logic signed [ACC-1:0] psum_in,  // from above
    output logic signed [DW-1:0]  a_out,    // to the right
    output logic signed [ACC-1:0] psum_out  // to below
);
    logic signed [DW-1:0] w;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            w        <= '0;
            a_out    <= '0;
            psum_out <= '0;
        end else begin
            if (w_load) w <= w_in;
            a_out    <= a_in;
            psum_out <= psum_in + ACC'(a_in * w);
        end
    end
endmodule
