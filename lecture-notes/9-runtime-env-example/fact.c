long fact(long n) { return n <= 1 ? 1 : n * fact(n - 1); }
long fact_acc(long n, long acc) { return n <= 1 ? acc : fact_acc(n - 1, n * acc); }
