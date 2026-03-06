// ============================================================
// DSA X — Domain-Specific Accelerator for Signal Processing
// Architecture: A → M → {N, O} → J
//
//   Module A : Unique input conditioner   (NAND / NOR)
//   Module M : Shared MAC unit            (AND / XOR / OR / DFF)
//   Module N : Shared data selector       (AND / NOT / MUX)
//   Module O : Data processor             (XOR / AND / MUX / BUF)
//   Module J : Output stage               (AND / BUF / DFF)
//
// M is *identical* in DSA_X, DSA_Y, DSA_Z  → shared component
// N is *identical* in DSA_X, DSA_Z         → shared component
// O is *similar* to Q (AND ↔ OR swap)      → mergeable pair
// J is *similar* to K (AND ↔ OR swap)      → mergeable pair
// A is unique to DSA_X
// ============================================================

// ── Top-level: A → M → {N, O} → J ──────────────────────────

module dsa_x_top(
    input  wire       clk,
    input  wire       rst,
    input  wire [3:0] data_in,
    input  wire [3:0] coeff,
    input  wire       sel,
    output wire [3:0] data_out
);
    wire [3:0] a_out;
    wire [7:0] m_acc;
    wire [3:0] n_out, o_out;

    input_cond_A   u_A (.in(data_in), .out(a_out));
    mac_unit_M     u_M (.clk(clk), .rst(rst), .a(a_out), .b(coeff), .acc(m_acc));
    data_sel_N     u_N (.in(m_acc[3:0]), .sel(sel), .out(n_out));
    data_proc_O    u_O (.in(m_acc[3:0]), .sel(sel), .out(o_out));
    output_stage_J u_J (.clk(clk), .rst(rst), .a(n_out), .b(o_out), .out(data_out));
endmodule


// ── Module A  (unique to DSA_X) ─────────────────────────────
// Gate pattern: NAND NAND NAND NAND  NOR NOR NOR NOR
module input_cond_A(
    input  wire [3:0] in,
    output wire [3:0] out
);
    wire [3:0] w;
    nand g0(w[0], in[0], in[1]);
    nand g1(w[1], in[1], in[2]);
    nand g2(w[2], in[2], in[3]);
    nand g3(w[3], in[3], in[0]);
    nor  g4(out[0], w[0], w[1]);
    nor  g5(out[1], w[1], w[2]);
    nor  g6(out[2], w[2], w[3]);
    nor  g7(out[3], w[3], w[0]);
endmodule


// ── Module M  (SHARED — identical in DSA_X, DSA_Y, DSA_Z) ──
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


// ── Module N  (shared in DSA_X and DSA_Z — identical code) ──
// Gate sequence: AND×2  NOT×2  MUX×4
module data_sel_N(
    input  wire [3:0] in,
    input  wire       sel,
    output wire [3:0] out
);
    wire [1:0] masked;
    wire [1:0] inverted;

    // Enable gating  (AND)
    and e0(masked[0], in[0], in[1]);
    and e1(masked[1], in[2], in[3]);

    // Inversion  (NOT)
    not i0(inverted[0], in[0]);
    not i1(inverted[1], in[1]);

    // Selection  (MUX via ternary — iverilog emits MUX functors)
    assign out[0] = sel ? masked[0]  : inverted[0];
    assign out[1] = sel ? masked[1]  : inverted[1];
    assign out[2] = sel ? in[2]      : masked[0];
    assign out[3] = sel ? in[3]      : masked[1];
endmodule


// ── Module O  (unique to DSA_X — similar to Q) ─────────────
// Gate sequence: XOR×3  AND×2  MUX×2  BUF×2
// Differs from Q: AND here  ↔  OR in Q  (mergeable pair)
module data_proc_O(
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

    // Masking  (AND)  ← differs from Q which uses OR
    and m0(gated[0], processed[0], in[0]);
    and m1(gated[1], processed[1], in[1]);

    // Routing  (MUX)
    assign out[0] = sel ? gated[0]    : processed[0];
    assign out[1] = sel ? gated[1]    : processed[1];

    // Drive  (BUF)
    buf b0(out[2], processed[2]);
    buf b1(out[3], in[3]);
endmodule


// ── Module J  (unique to DSA_X — similar to K) ─────────────
// Gate sequence: AND×3  BUF×2  DFF
// Differs from K: AND here  ↔  OR in K  (mergeable pair)
module output_stage_J(
    input  wire       clk, rst,
    input  wire [3:0] a, b,
    output reg  [3:0] out
);
    wire [2:0] merged;
    wire [1:0] driven;

    // Combine  (AND)  ← differs from K which uses OR
    and g0(merged[0], a[0], b[0]);
    and g1(merged[1], a[1], b[1]);
    and g2(merged[2], a[2], b[2]);

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
