using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Net;
using System.Text;

namespace AI
{
    class Program
    {
        public static HttpListener listener;
        public static VectorBuilder<double> _v = Vector<double>.Build;
        static void Main(string[] args)
        {

            #region
            //List<int> structure = new() { 2, 2, 4 };

            //// First output: xor
            //// Second output: xnor
            //// Third output: and
            //// Fourth output: or
            //Vector<double>[,] patterns =
            //{
            //    { _v.DenseOfArray(new double[] { 0, 1 }), _v.DenseOfArray(new double[] { 1, 0, 0, 1 }) },
            //    { _v.DenseOfArray(new double[] { 1, 0 }), _v.DenseOfArray(new double[] { 1, 0, 0, 1 }) },
            //    { _v.DenseOfArray(new double[] { 1, 1 }), _v.DenseOfArray(new double[] { 0, 1, 1, 1 }) },
            //    { _v.DenseOfArray(new double[] { 0, 0 }), _v.DenseOfArray(new double[] { 0, 1, 0, 0 }) },
            //};

            ////NeuralNetwork nn = new("D:/Users/zolix/Downloads/export.nns");
            //NeuralNetwork nn = new(structure, NeuralNetwork.ActivationType.TANH);

            //Stopwatch sw = Stopwatch.StartNew();
            ////nn.BackPropagateOffline(patterns, .2, 0, 2000);
            //nn.BackPropagateOnline(patterns, .2, 2, .2, 100000);
            //sw.Stop();
            //Console.WriteLine(sw.ElapsedMilliseconds);

            //Console.WriteLine("Outputs\n");

            //for (int i = 0; i < patterns.GetLength(0); i++)
            //{
            //    nn.Update(patterns[i, 0]);
            //    Console.WriteLine("Input:");
            //    for (int j = 0; j < patterns[i, 0].Count; j++)
            //    {
            //        Console.WriteLine(patterns[i, 0][j]);
            //    }
            //    Console.WriteLine("Output:");
            //    for (int j = 0; j < nn.Outputs.Count; j++)
            //    {
            //        Console.WriteLine(nn.Outputs[j]);
            //    }
            //    Console.WriteLine();
            //}
            #endregion

            #region
            //for (int i = 0; i < 1; i += 1)
            //{
            //    Thread thr1 = new Thread(() => Run(i));
            //    //Thread thr2 = new Thread(() => Run(i+1));
            //    thr1.Start();
            //    //thr2.Start();

            //    thr1.Join();
            //    //thr2.Join();

            //    Console.WriteLine(i + " finished");
            //}

            //

            //Vector<double>[][] patterns =
            //{
            //    new Vector<double>[] { _v.DenseOfArray(new double[] { 0, 1 }), _v.DenseOfArray(new double[] { 1, 0, 0, 1 }) },
            //    new Vector<double>[] { _v.DenseOfArray(new double[] { 1, 0 }), _v.DenseOfArray(new double[] { 1, 0, 0, 1 }) },
            //    new Vector<double>[] { _v.DenseOfArray(new double[] { 1, 1 }), _v.DenseOfArray(new double[] { 0, 1, 1, 1 }) },
            //    new Vector<double>[] { _v.DenseOfArray(new double[] { 0, 0 }), _v.DenseOfArray(new double[] { 0, 1, 0, 0 }) }
            //};

            //NeuralNetwork nn = new(new List<int> { 2, 3, 4 }, new List<NeuralNetwork.Layer.ActivationType> { NeuralNetwork.Layer.ActivationType.LINEAR, NeuralNetwork.Layer.ActivationType.SIGMOID, NeuralNetwork.Layer.ActivationType.SIGMOID }, NeuralNetwork.LossType.SQUAREERROR);

            //nn.BackPropagateOffline(patterns, .25, NeuralNetwork.Metrics.ERROR, double.NegativeInfinity, 500);

            //for (int i = 0; i < patterns.GetLength(0); i++)
            //{
            //    nn.Update(patterns[i][0]);
            //    Console.WriteLine("Input:");
            //    for (int j = 0; j < patterns[i].GetLength(0); j++)
            //    {
            //        Console.WriteLine(patterns[i][0][j]);
            //    }
            //    Console.WriteLine("Output:");
            //    for (int j = 0; j < nn.Outputs.Count; j++)
            //    {
            //        Console.WriteLine(nn.Outputs[j]);
            //    }
            //    Console.WriteLine();
            //}
            //
            #endregion



            Run(2);

            //HttpServer(new string[2] { "http://localhost:9463/", "http://192.168.0.16:9463/" });
        }

        public static void HttpServer(string[] prefixes)
        {
            NeuralNetwork network = new(@"C:\asd\90.nns");

            if (!HttpListener.IsSupported)
            {
                throw new Exception();
            }
            // URI prefixes are required,
            // for example "http://contoso.com:8080/index/".
            if (prefixes == null || prefixes.Length == 0)
                throw new ArgumentException("prefixes");

            // Create a listener.
            listener = new HttpListener();
            // Add the prefixes.
            foreach (string s in prefixes)
            {
                listener.Prefixes.Add(s);
            }
            listener.Start();
            Console.WriteLine("Listening...");
            while (true)
            {
                // Note: The GetContext method blocks while waiting for a request.
                HttpListenerContext context = listener.GetContext();
                HttpListenerRequest request = context.Request;

                Console.WriteLine(request.Url.ToString().Substring(34, request.Url.ToString().Length - 34 - 1).Split(','));

                double[] inputsArr = Array.ConvertAll(request.Url.ToString().Substring(34, request.Url.ToString().Length - 34 - 1).Split(','), element => double.Parse(element, CultureInfo.InvariantCulture));

                for (int i = 0; i < 28; i++)
                {
                    for (int j = 0; j < 28; j++)
                    {
                        Console.Write(inputsArr[28 * i + j] + " | ");
                    }
                    Console.WriteLine();
                }

                Vector<double> inputs = _v.DenseOfArray(inputsArr);

                // Obtain a response object.
                HttpListenerResponse response = context.Response;
                // Construct a response.
                string responseString = network.Update(inputs).MaximumIndex().ToString();
                //string responseString = "asd";
                byte[] buffer = Encoding.UTF8.GetBytes(responseString);
                // Get a response stream and write the response to it.
                response.ContentLength64 = buffer.Length;
                Stream output = response.OutputStream;
                output.Write(buffer, 0, buffer.Length);
                // You must close the output stream.
                output.Close();
            }
        }

        public static void Run(int num)
        {
            IEnumerable<Image> mnistTrainingSet = MnistReader.ReadTrainingData();

            //Console.WriteLine("Readed training set");

            Vector<double>[][] mnistFormattedTrainingSet = new Vector<double>[mnistTrainingSet.Count()][];

            int counter = 0;
            foreach (var image in mnistTrainingSet)
            {
                mnistFormattedTrainingSet[counter] = new Vector<double>[2];

                // Flatten array
                mnistFormattedTrainingSet[counter][0] = _v.DenseOfArray(image.Data.Cast<double>().ToArray()) / 255;

                Vector<double> expectedOutput = _v.Dense(10);
                expectedOutput[image.Label] = 1.0;

                mnistFormattedTrainingSet[counter][1] = expectedOutput;

                counter++;
            }

            //Console.WriteLine("Formatted training set");

            //Random rnd = new();

            NeuralNetwork nn = new(new List<uint> { 784, 128, 10 }, new List<NeuralNetwork.Layer.ActivationType> { NeuralNetwork.Layer.ActivationType.LINEAR, NeuralNetwork.Layer.ActivationType.LEAKYRELU, NeuralNetwork.Layer.ActivationType.SOFTMAX });

            //nn.GetAccuracy(mnistFormattedTrainingSet);

            //Console.ReadLine();

            //Console.WriteLine(nn.Update(mnistFormattedTrainingSet[100, 0]));

            nn.BackPropagateOnline(mnistFormattedTrainingSet, .001, 100, NeuralNetwork.LossType.SQUAREERROR, NeuralNetwork.Metrics.ACCURACY, double.PositiveInfinity, 1);

            //Console.WriteLine(nn.Update(mnistFormattedTrainingSet[100, 0]));

            //Console.WriteLine("Backpropagation Finished");

            Console.WriteLine($"Training set size: {counter}");

            mnistTrainingSet = MnistReader.ReadTestData();

            //Console.WriteLine("Readed testing set");

            int success = 0;
            counter = 0;
            foreach (var image in mnistTrainingSet)
            {
                Vector<double> outputs = nn.Update(_v.DenseOfArray(image.Data.Cast<double>().ToArray()) / 255);
                int result = outputs.MaximumIndex();

                //Console.WriteLine(outputs);
                //Console.WriteLine(image.Label);
                //Console.WriteLine(result);
                //Console.WriteLine();

                if (result == image.Label)
                {
                    success++;
                }

                counter++;
            }

            Console.WriteLine($"Test set size: {counter}");

            Console.WriteLine(num + ": " + counter + "/" + success);
            //Console.WriteLine("Finished");
            nn.Export("c:/asd/export" + num + ".nns");
        }

        static void OnProcessExit(object sender, EventArgs e)
        {
            listener.Stop();
        }
    }
}