
using NumSharp;

using System.Runtime.CompilerServices;

namespace AutoGrad
{
    public abstract class Function
    {
        public Function(params Tensor[] tensors)
        {
            Parents = tensors;
            requires_grad = Parents.Any( x => x?.RequiresGrad ?? false  );
        }

        public abstract NDArray Forward(params NDArray[] tensors);
        public abstract ITuple Backward(NDArray gradient);



        public Tensor?[] Parents { get; init; }
        protected bool requires_grad { get; init; }
        
        public static Tensor Apply<T>(params Tensor[] tensors)
            where T : Function
        {
            T? ctx = Activator.CreateInstance(typeof(T), tensors) as T;

            Tensor ret = new Tensor(ctx!
                .Forward(tensors
                .Select(i => i?.Numpy())
                .ToArray() ) , ctx.requires_grad);

            if (ctx.requires_grad && !Tensor.no_grad)
                ret.Context = ctx;

            return ret;
        }
#if false
        public static Tensor ApplyV2(Function ctx, params Tensor[] tensors)
        {
            Tensor ret = new Tensor(ctx!.Forward(tensors), ctx.requires_grad);

            if (ctx.requires_grad && !Tensor.no_grad)
                ret.Context = ctx;

            return ret;
        }
#endif
    }

    public class Add : Function
    {
        public Add(Tensor a, Tensor b)
            : base( a, b) {
;
        }

        public override NDArray Forward(params NDArray[] inputs)
        {
            return inputs[0] + inputs[1];
        }
       public override ITuple Backward(NDArray gradient)
        {
            return ( Parents[0]!.RequiresGrad ? Funcs.ReverseBroadCast(Parents[0]!.Numpy(), gradient ): null ,
                    Parents[1]!.RequiresGrad ? Funcs.ReverseBroadCast(Parents[1]!.Numpy(),gradient) : null );
        }

    }

    public class Subtract : Function
    {
        public Subtract(Tensor a, Tensor b)
            : base(a, b) {

        }

        public override NDArray Forward(params NDArray[] inputs)
        {
            return inputs[0] - inputs[1];
        }
       public override ITuple Backward(NDArray gradient)
        {
            return ( Parents[0]!.RequiresGrad ? Funcs.ReverseBroadCast(Parents[0]!.Numpy(),gradient) : null ,
                    Parents[1]!.RequiresGrad ? Funcs.ReverseBroadCast(Parents[1]!.Numpy(),- gradient) : null );
        }

    }

    public class Multiply : Function
    {
        public Multiply(Tensor a, Tensor b)
            : base(a, b) {
     ;
        }

        public override NDArray Forward(params NDArray[] inputs)
        {
            x = inputs[0]; 
            y = inputs[1];
            NDArray ret = x * y;
            if (!requires_grad)
            {
                x = null;
                y = null;
            }
            return ret;
        }
       public override ITuple Backward(NDArray gradient)
        {
            return ( Parents[0]!.RequiresGrad ? Funcs.ReverseBroadCast(x, y * gradient) : null ,
                    Parents[1]!.RequiresGrad ? Funcs.ReverseBroadCast(y, x *gradient) : null );
        }

        private NDArray? x, y;
    }


    public class Divide : Function
    {
        public Divide(Tensor a, Tensor b)
            : base(a, b) {
        }

        public override NDArray Forward(params NDArray[] inputs)
        {
            x = inputs[0];
            y = inputs[1];
            NDArray ret = x / y;
            if (!requires_grad)
            {
                x = null;
                y = null;
            }
            return ret;
        }
       public override ITuple Backward(NDArray gradient)
        {
            return ( Parents[0]!.RequiresGrad ? Funcs.ReverseBroadCast(x,gradient / y) : null ,
                    Parents[1]!.RequiresGrad ? Funcs.ReverseBroadCast(y,( gradient * -1) * x / (y*y))  : null );
        }

        private NDArray? x, y;
    }

    public class Pow : Function
    {
        public Pow(Tensor a, Tensor b)
            : base(a, b) {

        }
        
        public override NDArray Forward(params NDArray[] inputs)
        {
            x = inputs[0];
            y = inputs[1];
            if (y.size > 1)
                throw new Exception("only accepting scalar values as exponents");

            NDArray ret = np.power(x, y.GetData()[0] as ValueType);
            this.ret = ret;
            if (!requires_grad)
            {
                x = null;
                y = null;
                this.ret = null;
            }
            return ret;
        }
       public override ITuple Backward(NDArray gradient)
        {
            return ( Parents[0]!.RequiresGrad ? gradient*(y * (ret/x)) : null ,
                    Parents[1]!.RequiresGrad ? gradient*(np.log(x)*ret)  : null );
        }

        private NDArray? x, y , ret;
    }

    public class Exp : Function
    {
        public Exp(Tensor x)
            : base(x) {

        }

        public override NDArray Forward(params NDArray[] inputs)
        {
            NDArray ret = np.exp(inputs[0]);

            if (requires_grad)
                this.ret = ret;
            return ret;
        }
       public override ITuple Backward(NDArray gradient)
        {
            return new ValueTuple<NDArray>(requires_grad ? gradient * ret : null );
        }

        private NDArray? ret;
    }

    public class Log : Function
    {
        public Log(Tensor x)
            : base(x) {

        }

        public override NDArray Forward(params NDArray[] inputs)
        {
            x = inputs[0];  
            NDArray ret = np.log(x);

            if (!requires_grad)
                x = null;
            return ret;
        }
       public override ITuple Backward(NDArray gradient)
        {
            return new ValueTuple<NDArray>( requires_grad ? gradient / x : null);
        }

        private NDArray? x;
    }

    public class Sqrt : Function
    {
        public Sqrt(Tensor x)
            : base(x) {

        }

        public override NDArray Forward(params NDArray[] inputs)
        {
            x = inputs[0];
            
            NDArray ret = np.sqrt(x);

            if (!requires_grad)
                x = null;
            return ret;
        }
       public override ITuple Backward(NDArray gradient)
        {
            return new ValueTuple<NDArray>(requires_grad ?gradient * (x * .5) : null );
        }

        private NDArray? x;
    }




    public class Linear : Function
    {
        public Linear(Tensor inputs, Tensor w)
            : base(inputs, w ) {

        }
        public Linear(Tensor inputs , Tensor w , Tensor bias )
            : base(inputs , w , bias ) {

        }

        public override NDArray Forward(params NDArray[] inputs)
        {
            x = inputs[0];
            w = inputs[1];
            bias = Parents.Length == 3 ? inputs[2] : null;
            
            NDArray ret = Funcs.TensorDot(x, w, new int[]{ -1, 0 } );

            if (bias is not null)
                ret += bias;

            if (!requires_grad)
            {
                x = null;
                w = null;
            }
            return ret;
        }
       public override ITuple Backward(NDArray gradient)
        {
            int[] axes = Enumerable
                .Range(0, gradient.ndim - 1)
                .ToArray();

            NDArray? dB=null;
            if (bias is not null && Parents[2]!.RequiresGrad)
            {
                dB = gradient;
                foreach (int i in axes)
                    dB = dB!.sum( i , dtype: dB.dtype , keepdims: true);
                dB = Funcs.Squeeze(dB);
            }
            return (Parents[0]!.RequiresGrad ? Funcs.TensorDot(gradient, w!.T , new[] {-1,0}) : null,
                    Parents[1]!.RequiresGrad ? Funcs.TensorDot( x! , gradient , new int[][] { axes, axes }) : null ,
                    dB );
        }

        private NDArray? x, w , bias;
    }


    public class ReLU : Function
    {
        public ReLU(Tensor self)
            : base(self) {
;
        }

        public override NDArray Forward(params NDArray[] inputs)
        {
            NDArray ret = np.clip(inputs[0], 0, inputs[0]);
            if (requires_grad)
                self = ret;
            return ret;
        }
       public override ITuple Backward(NDArray gradient)
        {
            
            return  new ValueTuple<NDArray>( self?.astype(NPTypeCode.Boolean) * gradient ?? null);
        }

        NDArray? self;
    }

    public class Reshape : Function
    {
        public Reshape(Tensor self , Tensor? Input_shape)
            : base(self) {

        }

        public override NDArray Forward(params NDArray[] inputs)
        {
            Shape InputShape = new Shape(inputs[1].astype(NPTypeCode.Int32).ToArray<int>());
            NDArray ret = inputs[0].reshape(InputShape);
            if (requires_grad)
                input_shape = inputs[0].Shape;
            return ret;
        }
       public override ITuple Backward(NDArray gradient)
        {

            return new ValueTuple<NDArray>(requires_grad? gradient.reshape((Shape)input_shape!) : null );
        }

        Shape? input_shape;
    }


    class Sum : Function
    {
        public Sum(Tensor self, Tensor axes)
           : base(self) 
        {
            this.axes = (int[])axes.astype(NPTypeCode.Int32).GetData().ToArray();
        } 

        public override NDArray Forward(params NDArray[] inputs)
        {
            NDArray ret = inputs[0];
            
            if(axes.Length == ret.ndim)
            {
                if (requires_grad)
                {
                    input_shape = inputs[0];
                    outshape = new Shape(1);
                }
                return ret.sum().reshape(1);
            }

            foreach (int i in axes)
                ret = ret.sum(i, dtype: ret.dtype, keepdims: true);
            if (requires_grad)
            {
                input_shape = inputs[0];
                outshape = ret.Shape;
            }
            int[] t_axes = axes.Select(i => i>=0 ? i : i + inputs[0].ndim).ToArray();
            int[] output_shape = inputs[0].shape
                .Zip(Enumerable.Range(0, inputs[0].ndim))
                .Where(t => !t_axes.Contains(t.Second))
                .Select(t => t.First)
                .ToArray();
            
            return ret.reshape( output_shape );
        }
        public override ITuple Backward(NDArray gradient)
        {

            return new ValueTuple<NDArray>(requires_grad ? np.broadcast_to(
                gradient.reshape((Shape)outshape) , input_shape): null);
        }

        NDArray? input_shape;
        Shape? outshape;
        int[] axes;
    }


    public class Max : Function
    {
        public Max(Tensor self, Tensor axes)
           : base(self)
        {
            this.axes = (int[])axes.astype(NPTypeCode.Int32).GetData().ToArray();
        }

        public override NDArray Forward(params NDArray[] inputs)
        {
            NDArray ret = inputs[0];

            if (axes.Length == ret.ndim)
            {
                ret = ret.max().reshape(1);
                if (requires_grad)
                {
                    x = inputs[0];
                    this.ret = ret;
                }
                return ret;
            }

            foreach (int i in axes)
                ret = ret.max(i, dtype: ret.dtype, keepdims: true);
            if (requires_grad)
            {
                x = inputs[0];
                this.ret = ret;
            }
            int[] t_axes = axes.Select(i => i >= 0 ? i : i + inputs[0].ndim).ToArray();
            int[] output_shape = inputs[0].shape
                .Zip(Enumerable.Range(0, inputs[0].ndim))
                .Where(t => !t_axes.Contains(t.Second))
                .Select(t => t.First)
                .ToArray();

            return ret.reshape(output_shape);

        }
        public override ITuple Backward(NDArray gradient)
        {
            NDArray ones = (x == np.broadcast_to(ret, x)).astype(NPTypeCode.Float);

            NDArray premmean = ones;
            foreach (int i in axes)
                premmean = premmean.sum(i, dtype: premmean.dtype, keepdims: true);

            premmean = np.broadcast_to(premmean, x);

            NDArray mean = ones / premmean;

            NDArray broadcasted_grad = np.broadcast_to(gradient.reshape(ret.Shape), x);

            return new ValueTuple<NDArray>(requires_grad ? mean * broadcasted_grad : null);
        }

        NDArray? x;
        NDArray? ret;
        int[] axes;
    }

}
