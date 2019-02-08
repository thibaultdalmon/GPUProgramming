plot 'signal.data' using 1 title 'Input Signal' with lines,'signal.data' using 2 title 'Filtered Signal Box' with lines, 'signal.data' using 3 title 'Filtered Signal Gaussian' with lines
set xlabel "t"
set ylabel "S(t)"
set terminal postscript eps enhanced color "Arial" 24
set out "signal.eps"
replot

