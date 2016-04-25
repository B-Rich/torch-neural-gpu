function seq_equal(a, b)
   return a:eq(b):sum(2):eq(a:size(2)):sum()
end
