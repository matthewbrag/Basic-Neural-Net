namespace MatrixMath
{
    public class Matrix
    {
        public int Rows { get; private init; }
        public int Columns { get; private init; }

        public Matrix T {
            get
            {
                return new Matrix(Columns, Rows).Map((row, col, val) => this[col, row]);
            } 
        }

        private double[,] _data;

        public Matrix(double [,] data)
        {
            Rows = data.GetLength(0);
            Columns = data.GetLength(1);

            _data = data;
        }

        public Matrix(int rows, int cols)
        {
            Rows = rows;
            Columns = cols;

            _data = new double[rows, cols];
        }

        public static Matrix Random(int rows, int cols)
        {
            Matrix matrix = new Matrix(rows, cols);

            var r = new Random();

            return matrix.Map(v => 2 * r.NextDouble() - 1);
        }

        public double this[int row, int col]
        {
            get
            {
                return _data[row, col];
            }
            set
            {
                _data[row, col] = value;
            }
        }

        public Matrix Map(Func<double, double> function)
        {
            var m = new Matrix(Rows, Columns);

            for (var i = 0; i < Rows; i++)
            {
                for (var j = 0; j < Columns; j++)
                {
                    m[i, j] = function(this[i, j]);
                }
            }

            return m;
        }

        public Matrix Map(Func<int, int, double, double> function)
        {
            var m = new Matrix(Rows, Columns);

            for (var i = 0; i < Rows; i++)
            {
                for (var j = 0; j < Columns; j++)
                {
                    m[i, j] = function(i, j, this[i, j]);
                }
            }

            return m;
        }

        public override string ToString()
        {
            var s = "";

            for (var i = 0; i < Rows; i++)
            {
                for (var j = 0; j < Columns; j++)
                {
                    s += this[i, j] + " ";
                }
                s += "\n";
            }

            return s;
        }

        public static Matrix operator +(Matrix a, Matrix b)
        {
            if (!AreDimensionsEqual(a, b))
                throw new ArgumentException("Invalid matrix dimensions");


            return a.Map((row, col, val) => val + b[row, col]);
        }

        public static Matrix operator *(Matrix a, Matrix b)
        {
            if (a.Columns != b.Rows)
                throw new ArgumentException("Invalid matrix dimensions");

            var m = new Matrix(a.Rows, b.Columns);

            for (var i = 0; i < a.Rows; i++)
            {
                for (var j = 0; j < b.Columns; j++)
                {
                    var t = 0.0;

                    for(var k = 0; k < a.Columns; k++)
                    {
                        t += a[i, k] * b[k, j];
                    }

                    m[i, j ] = t;
                }
            }

            return m;
        }

        public static Matrix operator *(Matrix a, double s)
        {
            return a.Map(v => v * s);
        }

        public static Matrix operator *(double s, Matrix a)
        {
            return a * s;
        }

        //hammard product. NOT bitwise and
        public static Matrix operator &(Matrix a, Matrix b)
        {
            if(!AreDimensionsEqual(a, b))
            {
                throw new ArgumentException("Invalid dimensions");
            }

            return a.Map((row, col, val) => val*b[row, col]);
        }

        public static Matrix operator -(Matrix a, Matrix b)
        {
            return a.Map((row, col, val) => val - b[row, col]);
        }

        public static bool AreDimensionsEqual(Matrix a, Matrix b)
        {
            return a.Rows == b.Rows && a.Columns == b.Columns;
        }
    }
}
