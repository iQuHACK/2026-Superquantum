OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
h q[1];
cp(pi/2) q[1], q[0];
h q[0];
swap q[0], q[1];
