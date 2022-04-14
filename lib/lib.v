module tanh(clk, en, I, O,n_en);
	parameter FLOAT_SIZE = 24;
    parameter INT_SIZE = 8;

    input wire clk;
    input wire en;
    input wire signed [INT_SIZE-1:-FLOAT_SIZE] I;
    output wire signed [INT_SIZE-1:-FLOAT_SIZE] O;
	output n_en ;
	cordictanh inst_tanh(.CLK(clk),.EN(en),.z(I),.out(O),.next_stage_en(n_en));
end

module cordictanh(CLK, EN, z, out,next_stage_en);
    parameter FLOAT_SIZE = 24;
    parameter INT_SIZE = 8;

    input wire CLK;
    input wire EN;
    input wire signed [INT_SIZE-1:-FLOAT_SIZE] z;
    output wire signed [INT_SIZE-1:-FLOAT_SIZE] out;
	output next_stage_en ;
    parameter MAX_ITERATION_DIV = 31;
    reg signed [INT_SIZE-1:-FLOAT_SIZE] x_;
    reg signed [INT_SIZE-1:-FLOAT_SIZE] y_;
    reg signed [INT_SIZE-1:-FLOAT_SIZE] z_;
    reg signed [4:0] i;
    wire signed [INT_SIZE-1:-FLOAT_SIZE] Z_UPDATE;
    atanh_LOOKUP LOOKUP(
        .index(i),
        .value(Z_UPDATE));
    reg div_en;
    reg IS_FIRST4;
    reg IS_FIRST13;
    reg IS_Z_ZERO;
	
    always @(posedge CLK)
    begin
        if (EN) //  Like Reset
        begin
            x_ <= 32'h01_000000;
            y_ <= 32'h00_000000;
            z_ <= z;
            i <= -3;
            div_en <= 1'b0;
            IS_FIRST4 <= 1'b1;
            IS_FIRST13 <= 1'b1;
            IS_Z_ZERO <= 1'b0;
        end
        else
        begin
            if (|z_)    //  z not zero
            begin
                z_ <= z_[INT_SIZE-1] ? z_ + Z_UPDATE : z_ - Z_UPDATE;
            
                if (i < 1)
                begin
                    x_ <= z_[INT_SIZE-1] ? x_ - y_ + (y_ >>> -(i-2)) : x_ + y_ - (y_ >>> -(i-2));
                    y_ <= z_[INT_SIZE-1] ? y_ - x_ + (x_ >>> -(i-2)) : y_ + x_ - (x_ >>> -(i-2));
                    i <= i + 1;
                end
                else if (i == 4)
                begin
                    x_ <= z_[INT_SIZE-1] ? x_ - (y_ >>> i) : x_ + (y_ >>> i);
                    y_ <= z_[INT_SIZE-1] ? y_ - (x_ >>> i) : y_ + (x_ >>> i);
                    if (IS_FIRST4)  IS_FIRST4 <= 1'b0;
                    else            i <= i + 1; 
                end
                else if (i == 13)
                begin
                    x_ <= z_[INT_SIZE-1] ? x_ - (y_ >>> i) : x_ + (y_ >>> i);
                    y_ <= z_[INT_SIZE-1] ? y_ - (x_ >>> i) : y_ + (x_ >>> i);
                    if (IS_FIRST13)  IS_FIRST13 <= 1'b0;
                    else            i <= i + 1;     
                end
                else if (i < 14)
                begin
                    x_ <= z_[INT_SIZE-1] ? x_ - (y_ >>> i) : x_ + (y_ >>> i);
                    y_ <= z_[INT_SIZE-1] ? y_ - (x_ >>> i) : y_ + (x_ >>> i);
                    i <= i + 1;
                end
                else if (i == 14)
                begin
                    div_en <= 1'b1;
                    i <= i + 1;
                end
                else
                begin
                    div_en <= 1'b0;
                    i <= i;
                end
            end
            else
            begin
                if (IS_Z_ZERO)
                    div_en <= 1'b0;
                else
                begin
                    IS_Z_ZERO <= 1'b1;
                    div_en <= 1'b1;
                end
                     
            end
        end
    end

    cordicdiv divider(
        .CLK(CLK),
        .EN(div_en),
        .y(y_),
        .x(x_),
        .out(out),
		.next_stage_en(next_stage_en)
    );
endmodule

module cordicdiv(CLK, EN, y, x, out,next_stage_en);
    parameter FLOAT_SIZE = 24;
    parameter INT_SIZE = 8;

    input wire CLK;
    input wire EN;
    input wire signed [INT_SIZE-1:-FLOAT_SIZE] y;
    input wire signed [INT_SIZE-1:-FLOAT_SIZE] x;
    output reg signed [INT_SIZE-1:-FLOAT_SIZE] out;
	output reg next_stage_en;
    parameter MAX_ITERATION = FLOAT_SIZE+1;
    reg signed [INT_SIZE-1:-FLOAT_SIZE] y_;
    reg signed [INT_SIZE-1:-FLOAT_SIZE] z_;
    reg [4:0] i;
    always @(posedge CLK)
    begin
        if (EN) //  Like Reset
        begin
            out <= 32'h00_000000;
            z_ <= 32'h00_000000;
            y_ <= y;
            i <= 5'b0000;
			next_stage_en <= 1'b0;
        end
        else
        begin
            if (i < MAX_ITERATION && |y_)
            begin
                y_ <= y_[INT_SIZE-1] ? y_ + (x >>> i) : y_ - (x >>> i);
                z_ <= y_[INT_SIZE-1] ? z_ - $signed(32'h01_000000 >> i) : z_ + $signed(32'h01_000000 >> i);
                i <= i + 1;
            end
			else begin
				next_stage_en <= 1'b1;
			end
            out <= z_;
        end   
    end
endmodule

module relu(
    input [31:0] I,
    output [31:0] O,
	input clk,
	input en,
	output n_en
    );
	always@( posedge clk) begin
		if(en) begin
			O <= (I[31] == 0)? I : 0;   //if the sign bit is high, send zero on the output else send the input
			n_en <= 1'b1;
		end
		else begin
			n_en <= 1'b0;
		end
	end
endmodule

module sigmoid(I,O,clk,en,n_en);
parameter FLOAT_SIZE = 24;
parameter INT_SIZE = 8;
input [INT_SIZE-1:-FLOAT_SIZE] I;
output reg [INT_SIZE-1:-FLOAT_SIZE] O;
output reg n_en;
input clk,en;
reg [INT_SIZE-1:-FLOAT_SIZE] exp;
reg [INT_SIZE-1:-FLOAT_SIZE] x;
reg [INT_SIZE-1:-FLOAT_SIZE] mod_x;

reg [INT_SIZE-1:-FLOAT_SIZE] i;
reg signed [2:0] j;
wire [INT_SIZE-1:-FLOAT_SIZE] w1;
wire [INT_SIZE-1:-FLOAT_SIZE] w2;
wire [INT_SIZE-1:-FLOAT_SIZE] w3;
wire [INT_SIZE-1:-FLOAT_SIZE] w4;
reg isData;
wire isend;
fadd add1(.A(mod_x),.B(32'h01_000000),.O(w1));
fdiv div1(.A(x),.B(w1),.O(w2),.clk(clk),.start(isData),.end(isend));
fadd add1(.A(w2),.B(32'h01_000000),.O(w3));
fmult mult(.A(32'h00_100000),.B(w3),.O(w4));
always @(posedge clk) begin
	if(en) begin
		x <= I;
		mod_x = {1'b0,I[INT_SIZE-2:-FLOAT_SIZE]};
		isData <= 1'b1;
		n_en <= 1'b0;
	end
	else if(isData) begin
		O<=w4;
		n_en <= 1'b1;
	end
	else begin
		isData <= 1'b0;
	end
end
endmodule

module fmult(A,B,O) ;
input [31:0]A;
input [31:0]B;
output [31:0]O;
qmult #(24,32) mult_inst( .a(A),.b(B),.c(O));
endmodule

module fdiv(A,B,O,clk,start,end) ;
input [31:0]A;
input [31:0]B;
output [31:0]O;
qdiv #(24,32) div_inst( .dividend,(A),.divisor(B),.quotient_out(O),.clk(clk),.start(start),.complete(end));
endmodule

module fadd(A,B,O) ;
input [31:0]A;
input [31:0]B;
output [31:0]O;
qadd #(24,32) add_inst( .a(A),.b(B),.c(O));
endmodule

