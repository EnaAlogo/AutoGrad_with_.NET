using AutoGrad;
using NumSharp;


public class Tensor : NDArray
{

    public Tensor(NDArray? data, bool requires_grad = true)
       : base( values: data?.astype(NPTypeCode.Float)?.GetData() ?? 
           new NDArray(NPTypeCode.Float,size:0).GetData() , shape : data?.Shape ?? default)
    {
    
        RequiresGrad = requires_grad;
    }
    public Tensor(Shape? shape , bool requires_grad = true)
        : base(new float[shape?.Size ?? 0], shape ?? new Shape(0))
    {
        RequiresGrad = requires_grad;
    }

    public NDArray Numpy()
    {
        return new NDArray(GetData(), shape);
    }
    public Tensor Detach()
    {
        return new Tensor(this, false);
    }


    public static Tensor operator +(Tensor a, Tensor b)
    {
        return Function.Apply<Add>(a, b);
    }
    public static Tensor operator -(Tensor a, Tensor b)
    {
        return Function.Apply<Subtract>(a, b);
    }
    public static Tensor operator *(Tensor a, Tensor b)
    {
        return Function.Apply<Multiply>(a, b);
    }
    public static Tensor operator /(Tensor a, Tensor b)
    {
        return Function.Apply<Divide>(a, b);
    }
    public static Tensor operator +(Tensor a, NDArray b)
    {
        return Function.Apply<Add>(a, new Tensor(b, false));
    }
    public static Tensor operator -(Tensor a, NDArray b)
    {
        return Function.Apply<Subtract>(a, new Tensor(b, false));
    }
    public static Tensor operator *(Tensor a, NDArray b)
    {
        return Function.Apply<Multiply>(a, new Tensor(b, false));
    }
    public static Tensor operator /(Tensor a, NDArray b)
    {
        return Function.Apply<Divide>(a, new Tensor(b, false));
    }
    public static Tensor operator^(Tensor a, Tensor b)
    {
        return Function.Apply<Pow>(a, b);
    }
    public static Tensor operator ^(Tensor a, NDArray b)
    {
        return Function.Apply<Pow>(a, new Tensor(b , false) );
    }
    public Tensor Sqrt()
    {
        return Function.Apply<Sqrt>(this);
    }
    public Tensor Log()
    {
        return Function.Apply<Log>(this);
    }
    public Tensor Exp()
    {
        return Function.Apply<Exp>(this);
    }
    public Tensor Sum(params int[] axes)
    {
        int[] in_Axes = axes;
        if (axes.Length == 0)
            in_Axes = Enumerable.Range(0, ndim).ToArray();
        return Function.Apply<Sum>(this, new Tensor(new NDArray(in_Axes), false));
    }
    public Tensor Max(params int[] axes)
    {
        int[] in_Axes = axes;
        if (axes.Length == 0)
            in_Axes = Enumerable.Range(0, ndim).ToArray();
        return Function.Apply<Max>(this, new Tensor(new NDArray(in_Axes), false));
    }
    public Tensor Mean(params int[] axes)
    {
        Tensor sum = Sum(axes);
        return sum / axes.Aggregate(1, (a, b) => a * Shape[b]);
    }

    public Tensor Flatten(bool is_using_batch = false)
    {
        int[] flatshape = is_using_batch ? new int[] { shape[0], Shape.Size / shape[0] } :
            new int[] { Shape.Size };
        return Function.Apply<Reshape>(this, new Tensor(data: flatshape, requires_grad: false));
    }

    public Tensor Reshape(params int[] shape)
    {
        return Function.Apply<Reshape>(this , new Tensor(data:shape , requires_grad: false));
    }

    public bool RequiresGrad { get; set; }

    public Function? Context { get; set; }

    public Tensor? Gradient { get; set; }

    public static bool no_grad = false;

    private void __backward_r(Tensor? node,
            HashSet<Tensor?> visited,
            ref List<Tensor?> nodes)
    {
        visited.Add(node);
        if (node?.Context is not null)
        {
            foreach (var child in node.Context.Parents)
                if (!visited.Contains(child))
                    __backward_r(child, visited, ref nodes);
            nodes.Add(node);
        }
    }
    private List<Tensor?> _backward()
    {
        List<Tensor?> ret = new();
        __backward_r(this, new HashSet<Tensor?>() , ref ret);
        return ret;
    }
  
    public void backward()
    {
        Gradient = new(np.ones(shape) , requires_grad: false);
        _backward()
            .ToArray()
            .Reverse()
            .Where(x => x?.Context?.Parents.Any(p => p?.RequiresGrad ?? false) ?? false)
            .ToList()
            .ForEach( T =>{
                var grads = T!.Context!.Backward(T.Gradient!.Numpy());
                Enumerable
                    .Range(0, grads.Length)
                    .Select(i => new Tensor(grads?[i] as NDArray, requires_grad: false))
                    .Zip(T.Context.Parents)
                    .ToList()
                    .ForEach( tup =>{
                        var ( g , t) = tup;
                        if (g is not null && (t?.RequiresGrad ?? false)  )
                            t.Gradient = t.Gradient is not null ? t.Gradient + g : g;
                    });
                T.Context = null;
            });
        GC.Collect();
    }
}
