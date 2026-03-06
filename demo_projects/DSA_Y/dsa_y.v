// ============================================================
// DSA Y — Domain-Specific Accelerator for Matrix Processing
// Architecture: B → M → Q → K
//
//   Module B : Unique input conditioner   (XNOR / BUF)
//   Module M : Shared MAC unit            (AND / XOR / OR / DFF)
//   Module Q : Data processor             (XOR / OR / MUX / BUF)
//   Module K : Output stage               (OR / BUF / DFF)
//
// M is *identical* in DSA_X, DSA_Y, DSA_Z  → shared component
// Q is *identical* in DSA_Y, DSA_Z         → shared component
// Q is *similar* to O (OR ↔ AND swap)      → mergeable pair
// K is *similar* to J (OR ↔ AND swap)      → mergeable pair
// B is unique to DSA_Y
// ============================================================

// ── Top-level: B → M → Q → K ───────────────────────────────

module dsa_y_top(
    input  wire       clk,
    input  wire       rst,
    input  wire [3:0] data_in,
    input  wire [3:0] coeff,
    input  wire       sel,
    output wire [3:0] data_out
);
    wire [3:0] b_out;
    wire [7:0] m_acc;
    wire [3:0] q_out;

    input_cond_B   u_B (.in(data_in), .out(b_out));
    mac_unit_M     u_M (.clk(clk), .rst(rst), .a(b_out), .b(coeff), .acc(m_acc));
    data_proc_Q    u_Q (.in(m_acc[3:0]), .sel(sel), .out(q_out));
    output_stage_K u_K (.clk(clk), .rst(rst), .a(q_out), .b(m_acc[7:4]), .out(data_out));
endmodule


// ── Module B  (unique to DSA_Y) ─────────────────────────────
// Gate pattern: XNOR XNOR XNOR XNOR  BUF BUF BUF BUF
module input_cond_B(
    input  wire [3:0] in,
    output wire [3:0] out
);
    wire [3:0] w;
    xnor g0(w[0], in[0], in[1]);
    xnor g1(w[1], in[1], in[2]);
    xnor g2(w[2], in[2], in[3]);
    xnor g3(w[3], in[3], in[0]);
    buf  g4(out[0], w[0]);
    buf  g5(out[1], w[1]);
    buf  g6(out[2], w[2]);
    buf  g7(out[3], w[3]);
endmodule


// ── Module M  (SHARED — identical to DSA_X and DSA_Z) ──────
// Gate sequence: AND×8  XOR×4  AND×4  OR×4  DFF
module mac_unit_M(
    input  wire       clk, rst,
    input  wire [3:0] a, b,
    output reg  [7:0] acc
);
    wire [3:0] pp0, pp1;
    wire [3:0] sum_bits, carry_bits, combined;

    // Partial products  (AND)
    and pp_a0(pp0[0], a[0], b[0]);
    and pp_a1(pp0[1], a[1], b[0]);
    and pp_a2(pp0[2], a[2], b[0]);
    and pp_a3(pp0[3], a[3], b[0]);
    and pp_b0(pp1[0], a[0], b[1]);
    and pp_b1(pp1[1], a[1], b[1]);
    and pp_b2(pp1[2], a[2], b[1]);
    and pp_b3(pp1[3], a[3], b[1]);

    // Sum  (XOR)
    xor s0(sum_bits[0], pp0[0], pp1[0]);
    xor s1(sum_bits[1], pp0[1], pp1[1]);
    xor s2(sum_bits[2], pp0[2], pp1[2]);
    xor s3(sum_bits[3], pp0[3], pp1[3]);

    // Carry generation  (AND)
    and c0(carry_bits[0], pp0[0], pp1[0]);
    and c1(carry_bits[1], pp0[1], pp1[1]);
    and c2(carry_bits[2], pp0[2], pp1[2]);
    and c3(carry_bits[3], pp0[3], pp1[3]);

    // Combine sum + carry  (OR)
    or r0(combined[0], sum_bits[0], carry_bits[0]);
    or r1(combined[1], sum_bits[1], carry_bits[1]);
    or r2(combined[2], sum_bits[2], carry_bits[2]);
    or r3(combined[3], sum_bits[3], carry_bits[3]);

    // Accumulate register  (DFF)
    always @(posedge clk or posedge rst) begin
        if (rst)
            acc <= 8'b0;
        else
            acc <= acc + {4'b0, combined};
    end
endmodule


// ── Module Q  (shared in DSA_Y and DSA_Z — identical code) ──
// Gate sequence: XOR×3  OR×2  MUX×2  BUF×2
// Differs from O: OR here  ↔  AND in O  (mergeable pair)
module data_proc_Q(
    input  wire [3:0] in,
    input  wire       sel,
    output wire [3:0] out
);
    wire [2:0] processed;
    wire [1:0] gated;

    // Processing  (XOR)
    xor p0(processed[0], in[0], in[1]);
    xor p1(processed[1], in[1], in[2]);
    xor p2(processed[2], in[2], in[3]);

    // Combining  (OR)  ← differs from O which uses AND
    or m0(gated[0], processed[0], in[0]);
    or m1(gated[1], processed[1], in[1]);

    // Routing  (MUX)
    assign out[0] = sel ? gated[0]    : processed[0];
    assign out[1] = sel ? gated[1]    : processed[1];

    // Drive  (BUF)
    buf b0(out[2], processed[2]);
    buf b1(out[3], in[3]);
endmodule


// ── Module K  (unique to DSA_Y — similar to J) ─────────────
// Gate sequence: OR×3  BUF×2  DFF
// Differs from J: OR here  ↔  AND in J  (mergeable pair)
module output_stage_K(
    input  wire       clk, rst,
    input  wire [3:0] a, b,
    output reg  [3:0] out
);
    wire [2:0] merged;
    wire [1:0] driven;

    // Combine  (OR)  ← differs from J which uses AND
    or g0(merged[0], a[0], b[0]);
    or g1(merged[1], a[1], b[1]);
    or g2(merged[2], a[2], b[2]);

    // Output drive  (BUF)
    buf d0(driven[0], merged[0]);
    buf d1(driven[1], merged[1]);

    // Output register  (DFF)
    always @(posedge clk or posedge rst) begin
        if (rst)
            out <= 4'b0;
        else
            out <= {merged[2], driven[1], driven[0], a[3] ^ b[3]};
    end
endmodule
