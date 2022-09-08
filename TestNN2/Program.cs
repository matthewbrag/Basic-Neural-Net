using MatrixMath;

var layers = new[] { 2, 20, 20, 5, 1 };

var parameters = InitParameters(layers);

var r = new Random();
for (int i = 0; i < 1000; i++)
{
    var x0 = r.NextDouble() * 2 - 1;
    var x1 = r.NextDouble() * 2 - 1;

    var y = Math.Sqrt(x0 * x0 + x1 * x1) > 0.5 ? 1.0 : 0;

    var input = new Matrix(new[,]
    {
        { x0 },
        { x1 },
    });

    var output = new Matrix(new[,]
    {
        { y }
    });

    var cache = Forward(parameters, input);
    var gradient = Backward(parameters, cache, output);

    parameters = Update(parameters, gradient);
}

var correct = 0;
var total = 100;

for (int i = 0; i < total; i++)
{
    var x0 = r.NextDouble() * 2 - 1;
    var x1 = r.NextDouble() * 2 - 1;

    var y = Math.Sqrt(x0 * x0 + x1 * x1) > 0.5 ? 1.0 : 0;

    var input = new Matrix(new[,]
    {
        { x0 },
        { x1 },
    });

    var output = new Matrix(new[,]
    {
        { y }
    });

    var (A, Z) = Forward(parameters, input);

    var guess = A[^1][0,0];

    guess = Math.Round(guess);

    if(guess == y)
    {
        correct++;
    }
}

Console.WriteLine($"{correct} out of {total}");

(Matrix[] W, Matrix[] b) InitParameters(int[] layerSizes)
{
    var W = new Matrix[layerSizes.Length];
    var b = new Matrix[layerSizes.Length];

    for (int i = 1; i < layerSizes.Length; i++)
    {
        W[i] = Matrix.Random(layerSizes[i], layerSizes[i - 1]) * 0.01;
        b[i] = new Matrix(layerSizes[i], 1);
    }

    return (W, b);
}

(Matrix[] A, Matrix[] Z) Forward((Matrix[] W, Matrix[] b) parameters, Matrix x)
{
    var (W, b) = parameters;

    var A = new Matrix[W.Length];
    var Z = new Matrix[W.Length];

    A[0] = x;

    for (var i = 1; i < W.Length; i++)
    {
        Z[i] = W[i] * A[i - 1] + b[i];
        A[i] = Z[i].Map(Sigmoid);
    }

    return (A, Z);
}

(Matrix[] dW, Matrix[] db) Backward((Matrix[] W, Matrix[] b) parameters, (Matrix[] A, Matrix[] Z) cache, Matrix y)
{
    var (W, b) = parameters;

    var (A, Z) = cache;

    var dW = new Matrix[W.Length];
    var db = new Matrix[b.Length];

    var n = new Matrix[b.Length];

    n[^1] = (A[^1] - y) & Z[^1].Map(SigmoidPrime);

    for (var l = A.Length - 2; l >= 1; l--)
    {
        n[l] = W[l + 1].T * n[l + 1] & Z[l].Map(SigmoidPrime);
    }

    for (var l = 1; l < A.Length; l++)
    {


        dW[l] = n[l] * A[l - 1].T;
        db[l] = n[l];
    }

    return (dW, db);
}

(Matrix[] W, Matrix[] b) Update((Matrix[] W, Matrix[] b) parameters, (Matrix[] dW, Matrix[] db) gradient, double rate = 0.01)
{
    var (W, b) = parameters;

    var (dW, db) = gradient;

    for (var i = 1; i < W.Length; i++)
    {
        W[i] = W[i] - rate * dW[i];
        b[i] = b[i] - rate * db[i];
    }

    return (W, b);
}


double Sigmoid(double v)
{
    return 1 / (1 + Math.Exp(-v));
}

double SigmoidPrime(double v)
{
    return Sigmoid(v) * (1 - Sigmoid(v));
}