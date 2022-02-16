using OpenCL;
using System.Diagnostics;

namespace AI
{
    internal class Gpu
    {
        //private Context context;
        //private Device device;
        //private Event event0;

        //private void CheckErr(ErrorCode err, string name)
        //{
        //    if (err != ErrorCode.Success)
        //    {
        //        Console.WriteLine("ERROR: " + name + " (" + err.ToString() + ")");
        //    }
        //}
        //private void ContextNotify(string errInfo, byte[] data, IntPtr cb, IntPtr userData)
        //{
        //    Console.WriteLine("OpenCL Notification: " + errInfo);
        //}

        //public void Setup()
        //{
        //    ErrorCode error;
        //    Platform[] platforms = Cl.GetPlatformIDs(out error);
        //    List<Device> devicesList = new List<Device>();

        //    CheckErr(error, "Cl.GetPlatformIDs");

        //    foreach (Platform platform in platforms)
        //    {
        //        string platformName = Cl.GetPlatformInfo(platform, PlatformInfo.Name, out error).ToString();
        //        Console.WriteLine("Platform: " + platformName);
        //        CheckErr(error, "Cl.GetPlatformInfo");
        //        //We will be looking only for GPU devices
        //        foreach (Device device in Cl.GetDeviceIDs(platform, DeviceType.Gpu, out error))
        //        {
        //            CheckErr(error, "Cl.GetDeviceIDs");
        //            Console.WriteLine("Device: " + device.ToString());
        //            devicesList.Add(device);
        //        }
        //    }

        //    if (devicesList.Count <= 0)
        //    {
        //        Console.WriteLine("No devices found.");
        //        return;
        //    }

        //    device = devicesList[0];

        //    if (Cl.GetDeviceInfo(device, DeviceInfo.ImageSupport,
        //              out error).CastTo<Bool>() == Bool.False)
        //    {
        //        Console.WriteLine("No image support.");
        //        return;
        //    }
        //    context = Cl.CreateContext(null, 1, new[] { device }, ContextNotify, IntPtr.Zero, out error);    //Second parameter is amount of devices
        //    CheckErr(error, "Cl.CreateContext");
        //}

        unsafe public void Dot()
        {
            //ErrorCode error;
            ////Load and compile kernel source code.
            //string programPath = System.Environment.CurrentDirectory.Substring(0, System.Environment.CurrentDirectory.Length - 16) + "kernel.cl";
            ////The path to the source file may vary

            //if (!File.Exists(programPath))
            //{
            //    Console.WriteLine("Program doesn't exist at path " + programPath);
            //    return;
            //}

            //string programSource = File.ReadAllText(programPath);

            //OpenCL.Net.Program program = Cl.CreateProgramWithSource(context, 1, new[] { programSource }, null, out error);
            //CheckErr(error, "Cl.CreateProgramWithSource");
            ////Compile kernel source
            //error = Cl.BuildProgram(program, 1, new[] { device }, string.Empty, null, IntPtr.Zero);
            //CheckErr(error, "Cl.BuildProgram");
            ////Check for any compilation errors
            //if (Cl.GetProgramBuildInfo(program, device, ProgramBuildInfo.Status, out error).CastTo<BuildStatus>()
            //    != BuildStatus.Success)
            //{
            //    CheckErr(error, "Cl.GetProgramBuildInfo");
            //    Console.WriteLine("Cl.GetProgramBuildInfo != Success");
            //    Console.WriteLine(Cl.GetProgramBuildInfo(program, device, ProgramBuildInfo.Log, out error));
            //    return;
            //}

            ////Create the required kernel (entry function)
            //Kernel kernel = Cl.CreateKernel(program, "dot", out error);
            //CheckErr(error, "Cl.CreateKernel");

            //int M = 5;
            //int N = 10;
            //int K = 15;

            //double[,] A = new double[M, K];
            //double[,] B = new double[K, N];
            //double[,] C = new double[M, N];

            //var queue = Cl.CreateCommandQueue(context, device, 0, out error);
            //CheckErr(error, "Cl.CreateCommandQueue");

            //var buffA = Cl.CreateBuffer(context, MemFlags.ReadOnly, M * K * Marshal.SizeOf(A), out error);
            //CheckErr(error, "Cl.CreateBuffer");
            //var buffB = Cl.CreateBuffer(context, MemFlags.ReadOnly, K * N * Marshal.SizeOf(B), out error);
            //CheckErr(error, "Cl.CreateBuffer");
            ////var buffB_TR = Cl.CreateBuffer(context, MemFlags.ReadOnly, N * K * Marshal.SizeOf(A), out error);
            //var buffC = Cl.CreateBuffer(context, MemFlags.ReadOnly, M * N * Marshal.SizeOf(C), out error);
            //CheckErr(error, "Cl.CreateBuffer");

            //Cl.EnqueueWriteBuffer(queue, buffA, Bool.True, (IntPtr)0, (IntPtr)(M * K * Marshal.SizeOf(A)), A, 0, null, out event0);
            //CheckErr(error, "Cl.EnqueueWriteBuffer");
            //Cl.EnqueueWriteBuffer(queue, buffA, Bool.True, (IntPtr)0, (IntPtr)(K * N * Marshal.SizeOf(B)), B, 0, null, out event0);
            //CheckErr(error, "Cl.EnqueueWriteBuffer");
            //Cl.EnqueueWriteBuffer(queue, buffA, Bool.True, (IntPtr)0, (IntPtr)(M * N * Marshal.SizeOf(C)), C, 0, null, out event0);
            //CheckErr(error, "Cl.EnqueueWriteBuffer");

            //Cl.SetKernelArg(kernel, 0, (IntPtr)sizeof(int), M);
            //CheckErr(error, "Cl.SetKernelArg");
            //Cl.SetKernelArg(kernel, 1, (IntPtr)sizeof(int), N);
            //CheckErr(error, "Cl.SetKernelArg");
            //Cl.SetKernelArg(kernel, 2, (IntPtr)sizeof(int), K);
            //CheckErr(error, "Cl.SetKernelArg");
            //Cl.SetKernelArg(kernel, 3, (IntPtr)sizeof(Mem), A);
            //CheckErr(error, "Cl.SetKernelArg");
            //Cl.SetKernelArg(kernel, 4, (IntPtr)sizeof(Mem), B);
            //CheckErr(error, "Cl.SetKernelArg");
            //Cl.SetKernelArg(kernel, 5, (IntPtr)sizeof(Mem), C);
            //CheckErr(error, "Cl.SetKernelArg");

            //Cl.ReleaseEvent(event0);

            //IntPtr TS = (IntPtr)32;
            //IntPtr[] local = { TS, TS };
            //IntPtr[] global = { (IntPtr)M, (IntPtr)N };

            //Cl.EnqueueNDRangeKernel(queue, kernel, 2, null, global, local, 1, new Event[] { event0 }, out event0);
            //CheckErr(error, "Cl.EnqueueNDRangeKernel");

            //Cl.EnqueueNDRangeKernel(queue, kernel, 2, null, global, local, 1, new Event[] { event0 }, out event0);
            //CheckErr(error, "Cl.EnqueueNDRangeKernel");

            //Cl.WaitForEvents(1, new[] { event0 });

            string programPath = Environment.CurrentDirectory.Substring(0, Environment.CurrentDirectory.Length - 16) + "kernel.cl";
            EasyCL cl = new EasyCL();
            cl.Accelerator = AcceleratorDevice.GPU;

            string kernel = File.ReadAllText(programPath);

            const int aX = 2000, aY = 2000, bX = 2000, bY = 2000; //x=number of lines, y = number of columns
            double[] a = new double[aX * aY];
            double[] b = new double[bX * bY];
            double[] c = new double[aX * bY]; // resulting matrix, with aX*bY dimensions
            int[] dimensions = new int[3] { aX, aY, bY };

            Random rand = new Random();
            for (int i = 0; i < aX; i++)
            {
                for (int j = 0; j < aY; j++)
                {
                    a[i * aY + j] = rand.NextDouble();
                }
            }
            for (int i = 0; i < bX; i++)
            {
                for (int j = 0; j < bY; j++)
                {
                    b[i * bY + j] = rand.NextDouble();
                }
            }

            cl.LoadKernel(kernel);
            //Pass the dimensions and matrix pointers:
            //cl.SetParameter(dimensions, a, b, c);
            //Each cell of the result will be computed at the same time

            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();
            //cl.Execute(aX * bY);
            cl.Invoke("MatrixMulti", 0, aX * bY, dimensions, a, b, c);
            stopwatch.Stop();
            Console.WriteLine(stopwatch.Elapsed.ToString());

            for (int i = 0; i < aX; i++)
            {
                for (int j = 0; j < bY; j++)
                {
                    Console.Write(c[i * bY + j] + " ");
                }
                Console.WriteLine();
            }
        }
    }
}
