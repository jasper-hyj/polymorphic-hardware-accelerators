// ============================================================
// DSA Z — Domain-Specific Accelerator for Convolution
// Architecture: C → R → {N, Q} → M
//
//   Module C : Unique input conditioner   (NOT / NAND)
//   Module R : Unique processing block    (XNOR / NOR)
//   Module N : Shared data selector       (AND / NOT / MUX) — same as DSA_X
//   Module Q : Shared data processor      (XOR / OR / MUX / BUF) — same as DSA_Y
//   Module M : Shared MAC unit            (AND / XOR / OR / DFF) — same as DSA_X/Y
//
// M is *identical* in DSA_X, DSA_Y, DSA_Z  → shared component
// N is *identical* in DSA_X and DSA_Z       → shared component
// Q is *identical* in DSA_Y and DSA_Z       → shared component
// C, R are unique to DSA_Z
// ============================================================

// ── Top-level: C → R → {N, Q} → M ──────────────────────────

module dsa_z_top(
    input  wire       clk,
    input  wire       rst,
    input  wire [3:0] data_in,
    input  wire [3:0] coeff,
    input  wire       sel,
    output wire [3:0] data_out
);
    wire [3:0] c_out, r_out;
    wire [3:0] n_out, q_out;
    wire [7:0] m_acc;

    input_cond_C  u_C (.in(data_in), .out(c_out));
    unique_proc_R u_R (.in(c_out), .out(r_out));
    data_sel_N    u_N (.in(r_out), .sel(sel), .out(n_out));
    data_proc_Q   u_Q (.in(r_out), .sel(sel), .out(q_out));
    mac_unit_M    u_M (.clk(clk), .rst(rst), .a(n_out), .b(q_out), .acc(m_acc));

    assign data_out = m_acc[3:0];
endmodule


// ── Module C  (unique to DSA_Z) ─────────────────────────────
// Gate pattern: NOT NOT NOT NOT  NAND NAND NAND NAND
module input_cond_C(
    input  wire [3:0] in,
    output wire [3:0] out
);
    wire [3:0] w;
    not  g0(w[0], in[0]);
    not  g1(w[1], in[1]);
    not  g2(w[2], in[2]);
    not  g3(w[3], in[3]);
    nand g4(out[0], w[0], w[1]);
    nand g5(out[1], w[1], w[2]);
    nand g6(out[2], w[2], w[3]);
    nand g7(out[3], w[3], w[0]);
endmodule


// ── Module R  (unique to DSA_Z) ─────────────────────────────
// Gate pattern: XNOR XNOR XNOR XNOR  NOR NOR NOR NOR
module unique_proc_R(
    input  wire [3:0] in,
    output wire [3:0] out
);
    wire [3:0] w;
    xnor g0(w[0], in[0], in[1]);
    xnor g1(w[1], in[1], in[2]);
    xnor g2(w[2], in[2], in[3]);
    xnor g3(w[3], in[3], in[0]);
    nor  g4(out[0], w[0], w[1]);
    nor  g5(out[1], w[1], w[2]);
    nor  g6(out[2], w[2], w[3]);
    nor  g7(out[3], w[3], w[0]);
endmodule


// ── Module N  (SHARED — identical to DSA_X) ─────────────────
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


// ── Module Q  (SHARED — identical to DSA_Y) ─────────────────
// Gate sequence: XOR×3  OR×2  MUX×2  BUF×2
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

    // Combining  (OR)
    or m0(gated[0], processed[0], in[0]);
    or m1(gated[1], processed[1], in[1]);

    // Routing  (MUX)
    assign out[0] = sel ? gated[0]    : processed[0];
    assign out[1] = sel ? gated[1]    : processed[1];

    // Drive  (BUF)
    buf b0(out[2], processed[2]);
    buf b1(out[3], in[3]);
endmodule


// ── Module M  (SHARED — identical to DSA_X and DSA_Y) ──────
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
