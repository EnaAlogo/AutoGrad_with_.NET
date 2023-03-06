
using NumSharp;

namespace AutoGrad{ 
    class Funcs
    {

        public static NDArray Squeeze(in NDArray a)
        {
            var s = a.shape
                .Where(i => i != 1)
                .ToList();

            if (!s.Any())
                s.Add(1);
            return a.reshape(s.ToArray());
        }
        public static NDArray TensorDot(in NDArray a, in NDArray b, int axes)
        {
            (int[] a_axes, int[] b_axes) = TensorDotAxes(a, axes);
            (NDArray a_reshape, int[] a_free_dims) = TensorDotReshape(a, a_axes);
            (NDArray b_reshape, int[] b_free_dims) = TensorDotReshape(b, b_axes ,true);
            var contraction = a_free_dims.Concat(b_free_dims);
            NDArray ab_matmul = np.matmul(a_reshape, b_reshape);
            if (!ab_matmul.shape.SequenceEqual(contraction))
                ab_matmul = ab_matmul.reshape(contraction.ToArray());
            return ab_matmul;
        }

        public static NDArray TensorDot(in NDArray a, in NDArray b, int[] axes)
        {
            (int a_axes, int b_axes) = TensorDotAxes(a, axes);
            (NDArray a_reshape, int[] a_free_dims) = TensorDotReshape(a, a_axes);
            (NDArray b_reshape, int[] b_free_dims) = TensorDotReshape(b, b_axes, true);
            var contraction = a_free_dims.Concat(b_free_dims);
            NDArray ab_matmul = np.matmul(a_reshape, b_reshape);
            if (!ab_matmul.shape.SequenceEqual(contraction))
                ab_matmul = ab_matmul.reshape(contraction.ToArray());
            return ab_matmul;
        }

        public static NDArray TensorDot(in NDArray a, in NDArray b, int[][] axes)
        {
            (int[] a_axes, int[] b_axes) = TensorDotAxes(a, axes);
            (NDArray a_reshape, int[] a_free_dims) = TensorDotReshape(a, a_axes);
            (NDArray b_reshape, int[] b_free_dims) = TensorDotReshape(b, b_axes, true);
            var contraction = a_free_dims.Concat(b_free_dims);
            NDArray ab_matmul = np.matmul(a_reshape, b_reshape);
            if (!ab_matmul.shape.SequenceEqual(contraction))
                ab_matmul = ab_matmul.reshape(contraction.ToArray());
            return ab_matmul;
        }


        static private (NDArray, int[]) TensorDotReshape(in NDArray a, int axes, bool flipped = false)
        {
            int[] shape = a.shape;

            int[] free = Enumerable
                .Range(0, a.ndim)
                .Where(i => i != axes)
                .ToArray();

            int[] free_dims = free
                .Select(i => shape[i])
                .ToArray();

            int prod_free = free
                .Aggregate(1, (a, b) => a * shape[b]);

            int prod_axes = a.shape[axes];

            int[] perms = free , new_shape;
            if (flipped)
            {
                perms = perms.Prepend(axes).ToArray();
                new_shape = new int[] { prod_axes, prod_free };
            }
            else
            {
                perms = perms.Append(axes).ToArray();
                new_shape = new int[] { prod_free, prod_axes };
            }
            int[] dims = Enumerable.Range(0, a.ndim).ToArray();

            NDArray a_trans = !perms.SequenceEqual(dims) ? np.transpose(a, perms) : a;
            return (a_trans.reshape(new_shape), free_dims);

        }
        static private (NDArray , int[]) TensorDotReshape(in NDArray a , int[] axes, bool flipped = false )
        {
            int[] shape = a.shape;

            int[] free = Enumerable
                .Range(0, a.ndim)
                .Where(i => !axes.Contains(i))
                .ToArray();

            int[] free_dims = free
                .Select(i => shape[i])
                .ToArray();

            int prod_free = free
                .Select( i => shape[i])
                .Aggregate(1, (a, b) => a*b);

            int prod_axes = axes
                .Select( i=>shape[i])
                .Aggregate(1, (a, b) => a*b);

            int[] perms, new_shape;
            if (flipped){
                perms = axes.Concat(free).ToArray();
                new_shape = new int[]{ prod_axes, prod_free};
            }
            else{
                perms = free.Concat(axes).ToArray();
                new_shape = new int[] { prod_free, prod_axes }; 
            }
            int[] dims = Enumerable.Range(0, a.ndim).ToArray();

            NDArray a_trans = !perms.SequenceEqual(dims) ? np.transpose(a, perms) : a;
            return (a_trans.reshape(new_shape), free_dims);

        }

        static private (int,int) TensorDotAxes(in NDArray a, in int[] axes)
        {
            if (axes.Length != 2)
                throw new Exception("invalid axes arg must be 2 in len");

            return (axes![0] >= 0 ? axes[0] : axes[0] + a.ndim ,
                    axes![1] >= 0 ? axes[1] : axes[1] + a.ndim);
        }

        static private (int[], int[]) TensorDotAxes(in NDArray a, int axes)
        {
            if (axes > a.ndim || axes < 0)
                throw new Exception("invalid axis arg must be (0,rank]");
            int range = a.ndim - axes;
            return (
                Enumerable.Range(range, axes).ToArray(),
                Enumerable.Range(0, axes).ToArray() );
        }
        
        static private (int[] , int[]) TensorDotAxes(in NDArray a,in int[][] axes)
        {
            if (axes[0].Length != axes[1].Length || axes.Length != 2)
                throw new Exception("axes must have same length and be 2");

            int dim = a.ndim;
            var filter = (int axis) => axis >= 0 ? axis : axis + dim;

            int[] a_axes = axes[0].Select(filter).ToArray();
            int[] b_axes = axes[1].Select(filter).ToArray();

            return (a_axes, b_axes);
        }


        static public NDArray ReverseBroadCast(in NDArray initial, in NDArray post)
        {
            NDArray ret = post;
            foreach (int i in ReverseBroadcastAxes(initial , post) )
                ret = ret.sum(i, dtype: post.dtype, keepdims: true);
            return ret.reshape(initial.Shape);
        }

        static public int[] ReverseBroadcastAxes (in NDArray initial, in NDArray resulting)
        {
            return
            Enumerable
                .Repeat(1, resulting.ndim - initial.ndim)
                .Concat(initial.shape)
                .Zip(resulting.shape , Enumerable.Range(0 , resulting.ndim))
                .Where(t => t.First != t.Second && t.Second != 1 )
                .Select( t=> t.Third)
                .ToArray();
        }


        static public NDArray ReverseBroadCast(in Shape initial, in NDArray post)
        {
            NDArray ret = post;
            foreach (int i in ReverseBroadcastAxes(initial, post))
                ret = ret.sum(i, dtype: post.dtype, keepdims: true);
            return ret.reshape(initial);
        }
        static public int[] ReverseBroadcastAxes(in Shape initial, in NDArray resulting)
        {
            return
            Enumerable
                .Repeat(1, resulting.ndim - initial.NDim)
                .Concat(initial.Dimensions)
                .Zip(resulting.shape, Enumerable.Range(0, resulting.ndim))
                .Where(t => t.First != t.Second && t.Second != 1)
                .Select(t => t.Third)
                .ToArray();
        }

        static public Tensor Linear(in Tensor x , in Tensor w , in Tensor? b = null)
        {
            return b is not null ? Function.Apply<AutoGrad.Linear>(x, w, b)
                : Function.Apply<AutoGrad.Linear>(x, w);
        }

    }
}
