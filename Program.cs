namespace AI
{
    class Program
    {
        static void Main(string[] args)
        {

            List<int> dimensions = new() { 2, 3, 1 };
            double[] inputs = { 0.0, 1.0 };
            var asd2 = new NeuralNetwork(dimensions, ActivationTypes.TANH, 0.1);
            asd2.BackPropagate().SetInputs(inputs).Update();
            Console.WriteLine(asd2.ToString());
        }
    }
}
