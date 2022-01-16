namespace AI
{
    public class NeuralNetwork
    {
        public List<Layer> Layers { get; set; }
        public double LearningRate;

        public List<double> GetInputs()
        {
            List<double> inputs = new();
            for (int i = 0; i < Layers[0].NeuronsNumber; i++)
            {
                inputs.Add(Layers[0].Outputs[i]);
            }
            return inputs;
        }

        public double GetInputs(int i)
        {

            return Layers[0].Outputs[i];
        }

        public NeuralNetwork SetInputs(List<double> inputs)
        {
            Layers[0].Outputs = inputs;

            return this;
        }

        public NeuralNetwork SetInputs(int i, double input)
        {
            Layers[0].Outputs[i] = input;

            return this;
        }

        public NeuralNetwork(List<Layer> layers = null, double learningRate = 1)
        {
            layers ??= new List<Layer>();
            Layers = layers;
            LearningRate = learningRate;
        }

        public NeuralNetwork(List<int> structure, ActivationTypes activationType, double learningRate = 1)
        {
            LearningRate = learningRate;

            List<ActivationTypes> _activationTypes = new();

            for (int i = 0; i < structure.Count; i++)
            {
                if (i == 0)
                {
                    _activationTypes.Add(ActivationTypes.LINEAR);
                }
                else
                {
                    _activationTypes.Add(activationType);
                }
            }

            Build(structure, _activationTypes);
        }

        public NeuralNetwork(List<int> structure, List<ActivationTypes> activationTypes, double learningRate = 1)
        {
            LearningRate = learningRate;

            Build(structure, activationTypes);
        }

        private void Build(List<int> structure, List<ActivationTypes> activationTypes)
        {
            if (activationTypes.Count != structure.Count)
            {
                throw new Exception("First dimension of activationTypes must be equal to the number of layers");
            }

            Layers = new List<Layer>();

            Layers.Add(new Layer(structure[0], 0, activationTypes[0]));

            for (int i = 1; i < structure.Count; i++)
            {
                Layers.Add(new Layer(structure[i], structure[i - 1], activationTypes[i]));
            }
        }

        public List<double> Update()
        {
            // Skip input layer
            for (int i = 1; i < Layers.Count; i++)
            {
                Layers[i].Update(Layers[i - 1].Outputs);
            }

            return Layers[Layers.Count - 1].Outputs;
        }

        public NeuralNetwork BackPropagate()
        {
            // Only trains for first pattern
            // TODO: you know..

            List<List<List<double>>> expectations = new()
            {
                new() { new() { 0, 0 }, new() { 0 } },
                new() { new() { 1, 1 }, new() { 0 } },
                new() { new() { 0, 1 }, new() { 1 } },
                new() { new() { 1, 0 }, new() { 1 } }
            };

            List<List<double>> outputs = new();
            for (int i = 0; i < expectations.Count; i++)
            {
                outputs.Add(SetInputs(expectations[i][0]).Update());
            }

            List<List<double>> errors = new();
            List<List<double>> dErrors = new();
            for (int i = 0; i < outputs.Count; i++)
            {
                errors.Add(new());
                dErrors.Add(new());
                for (int j = 0; j < outputs[i].Count; j++)
                {
                    errors[i].Add(Math.Pow(outputs[i][j] - expectations[i][1][j], 2) / 2);
                    dErrors[i].Add(outputs[i][j] - expectations[i][1][j]);
                }
            }

            for (int i = Layers.Count - 1; i >= 0; i--)
            {
                Layers[i].WeightsErrors.Clear();
                Layers[i].NodesErrors.Clear();

                for (int j = 0; j < Layers[i].NeuronsNumber; j++)
                {
                    if (i == Layers.Count - 1)
                    {
                        Layers[i].NodesErrors.Add(dErrors[0][j] * Layers[i].Derivatives[j]);
                    }
                    else
                    {
                        double sum = 0;

                        for (int k = 0; k < Layers[i + 1].NeuronsNumber; k++)
                        {
                            sum += Layers[i + 1].NodesErrors[k] * Layers[i + 1].Weights[k][j];
                        }

                        Console.WriteLine(sum * Layers[i].Derivatives[j]);
                        Layers[i].NodesErrors.Add(sum * Layers[i].Derivatives[j]);
                    }

                    if (i != 0)
                    {
                        Layers[i].WeightsErrors.Add(new());

                        for (int k = 0; k < Layers[i].LastLayerNeuronsNumber; k++)
                        {
                            Layers[i].WeightsErrors[j].Add(Layers[i].NodesErrors[j] * Layers[i - 1].Outputs[k]);
                        }
                    }
                }
            }

            for (int i = 1; i < Layers.Count; i++)
            {
                for (int j = 0; j < Layers[i].NeuronsNumber; j++)
                {
                    for (int k = 0; k < Layers[i].LastLayerNeuronsNumber; k++)
                    {
                        Layers[i].Weights[j][k] -= Layers[i].WeightsErrors[j][k] * LearningRate;
                    }
                }
            }

            return this;
        }

        public override string ToString()
        {
            string str = "";

            foreach (Layer layer in Layers)
            {
                str += layer + "\n\n";
            }
            return str;
        }
    }

    public class Layer
    {
        public List<List<double>> Weights { get; set; } = new();
        public List<double> Biases { get; set; } = new();
        public List<double> Outputs { get; set; } = new();
        public List<double> Derivatives { get; set; } = new();
        public List<List<double>> WeightsErrors { get; set; } = new();
        public List<double> NodesErrors { get; set; } = new();
        ActivationTypes ActivationType { get; set; }
        public int NeuronsNumber
        {
            get
            {
                return Weights.Count;
            }
        }
        public int LastLayerNeuronsNumber
        {
            get
            {
                return Weights[0].Count;
            }
        }

        public Layer(int neuronsNumber, int lastLayerNeuronsNumber, ActivationTypes activationType)
        {
            Random random = new Random();

            for (int i = 0; i < neuronsNumber; i++)
            {
                Weights.Add(new List<double>());
                Biases.Add(random.NextDouble());
                for (int j = 0; j < lastLayerNeuronsNumber; j++)
                {
                    Weights[i].Add(random.NextDouble());
                }
            }

            ActivationType = activationType;
        }

        public List<double> Update(List<double> inputs)
        {
            if (inputs.Count != LastLayerNeuronsNumber)
            {
                throw new Exception("Inputs number must be equal to LastLayerNeuronsNumber");
            }

            Outputs.Clear();
            Derivatives.Clear();

            for (int i = 0; i < Weights.Count; i++)
            {
                Outputs.Add(0);
                for (int j = 0; j < Weights[i].Count; j++)
                {
                    Outputs[i] += inputs[j] * Weights[i][j];
                }
                Outputs[i] += Biases[i];

                Outputs[i] = Activate(Outputs[i]);

                Derivatives.Add(Outputs[i] * (1 - Outputs[i]));
            }

            return Outputs;
        }

        public double Activate(double value)
        {
            switch (ActivationType)
            {
                case ActivationTypes.BINARYSTEP: return Activation.BinaryStep.Default(value);
                case ActivationTypes.LINEAR: return Activation.Linear.Default(value);
                case ActivationTypes.SIGMOID: return Activation.Sigmoid.Default(value); ;
                case ActivationTypes.TANH: return Activation.TanH.Default(value);
                case ActivationTypes.RELU: return Activation.ReLU.Default(value);
                default: throw new NotImplementedException();
            }
        }

        public override string ToString()
        {
            string str = "";

            for (int i = 0; i < NeuronsNumber; i++)
            {
                string _weights = "(";
                for (int j = 0; j < Weights[i].Count; j++)
                {
                    _weights += $"{Weights[i][j]}, ";
                }
                _weights += ")";

                str += $"({_weights}, {Biases[i]}, {Outputs[i]})\n";
            }

            return str;
        }
    }

    public enum ActivationTypes
    {
        BINARYSTEP,
        LINEAR,
        SIGMOID,
        TANH,
        RELU
    }

    // TODO: implement all derivatives
    public class Activation
    {
        public static class BinaryStep
        {
            public static int Default(double value)
            {
                return Convert.ToInt32(value >= 0);
            }

            public static double Derivative(double value)
            {
                throw new NotImplementedException();
            }
        }

        public static class Linear
        {
            public static double Default(double value)
            {
                return value;
            }

            public static double Derivative(double value)
            {
                throw new NotImplementedException();
            }
        }

        public static class Sigmoid
        {
            public static double Default(double value)
            {
                return 1.0f / (1.0f + (float)Math.Exp(-value));
            }

            public static double Derivative(double value)
            {
                return Math.Exp(-value) / Math.Pow(1 + Math.Exp(-value), 2.0f);
            }
        }

        public static class TanH
        {
            public static double Default(double value)
            {
                return Math.Tanh(value);
            }

            public static double Derivative(double value)
            {
                throw new NotImplementedException();
            }
        }

        public static class ReLU
        {
            public static double Default(double value)
            {
                return Math.Max(0, value);
            }

            public static double Derivative(double value)
            {
                throw new NotImplementedException();
            }
        }
    }

    public class Train
    {
        public static int localMinUtil(List<double> arr, int low, int high, int n)
        {

            // Find index of middle element
            int mid = low + (high - low) / 2;

            // Compare middle element with its neighbours
            // (if neighbours exist)
            if (mid == 0 || arr[mid - 1] > arr[mid] &&
               mid == n - 1 || arr[mid] < arr[mid + 1])
                return mid;

            // If middle element is not minima and its left
            // neighbour is smaller than it, then left half
            // must have a local minima.
            else if (mid > 0 && arr[mid - 1] < arr[mid])
                return localMinUtil(arr, low, mid - 1, n);

            // If middle element is not minima and its right
            // neighbour is smaller than it, then right half
            // must have a local minima.
            return localMinUtil(arr, mid + 1, high, n);
        }

        public static int localMinUtil2(List<double> arr, int low, int high, int n)
        {

            // Find index of middle element
            int mid = low + (high - low) / 2;

            // Compare middle element with its neighbours
            // (if neighbours exist)
            if (mid == 0 || arr[mid - 1] > arr[mid] &&
               mid == n - 1 || arr[mid] < arr[mid + 1])
                return mid;

            // If middle element is not minima and its left
            // neighbour is smaller than it, then left half
            // must have a local minima.
            else if (mid > 0 && arr[mid - 1] < arr[mid])
                return localMinUtil(arr, low, mid - 1, n);

            // If middle element is not minima and its right
            // neighbour is smaller than it, then right half
            // must have a local minima.
            return localMinUtil(arr, mid + 1, high, n);
        }

        // A wrapper over recursive function localMinUtil()
        public static int localMin(List<double> arr, int n)
        {
            return localMinUtil(arr, 0, n - 1, n);
        }
    }
}