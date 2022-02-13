using OpenCL.Net;
using System.Runtime.InteropServices;

namespace AI
{
    internal class Gpu
    {
        private Context _context;
        private Device _device;

        private void CheckErr(ErrorCode err, string name)
        {
            if (err != ErrorCode.Success)
            {
                Console.WriteLine("ERROR: " + name + " (" + err.ToString() + ")");
            }
        }
        private void ContextNotify(string errInfo, byte[] data, IntPtr cb, IntPtr userData)
        {
            Console.WriteLine("OpenCL Notification: " + errInfo);
        }

        public void Setup()
        {
            ErrorCode error;
            Platform[] platforms = Cl.GetPlatformIDs(out error);
            List<Device> devicesList = new List<Device>();

            CheckErr(error, "Cl.GetPlatformIDs");

            foreach (Platform platform in platforms)
            {
                string platformName = Cl.GetPlatformInfo(platform, PlatformInfo.Name, out error).ToString();
                Console.WriteLine("Platform: " + platformName);
                CheckErr(error, "Cl.GetPlatformInfo");
                //We will be looking only for GPU devices
                foreach (Device device in Cl.GetDeviceIDs(platform, DeviceType.Gpu, out error))
                {
                    CheckErr(error, "Cl.GetDeviceIDs");
                    Console.WriteLine("Device: " + device.ToString());
                    devicesList.Add(device);
                }
            }

            if (devicesList.Count <= 0)
            {
                Console.WriteLine("No devices found.");
                return;
            }

            _device = devicesList[0];

            if (Cl.GetDeviceInfo(_device, DeviceInfo.ImageSupport,
                      out error).CastTo<Bool>() == Bool.False)
            {
                Console.WriteLine("No image support.");
                return;
            }
            _context = Cl.CreateContext(null, 1, new[] { _device }, ContextNotify, IntPtr.Zero, out error);    //Second parameter is amount of devices
            CheckErr(error, "Cl.CreateContext");
        }

        unsafe public void Dot()
        {
            ErrorCode error;
            //Load and compile kernel source code.
            string programPath = System.Environment.CurrentDirectory + "/kernel.cl";
            //The path to the source file may vary

            if (!File.Exists(programPath))
            {
                Console.WriteLine("Program doesn't exist at path " + programPath);
                return;
            }

            string programSource = File.ReadAllText(programPath);

            OpenCL.Net.Program program = Cl.CreateProgramWithSource(_context, 1, new[] { programSource }, null, out error);
            CheckErr(error, "Cl.CreateProgramWithSource");
            //Compile kernel source
            error = Cl.BuildProgram(program, 1, new[] { _device }, string.Empty, null, IntPtr.Zero);
            CheckErr(error, "Cl.BuildProgram");
            //Check for any compilation errors
            if (Cl.GetProgramBuildInfo(program, _device, ProgramBuildInfo.Status, out error).CastTo<BuildStatus>()
                != BuildStatus.Success)
            {
                CheckErr(error, "Cl.GetProgramBuildInfo");
                Console.WriteLine("Cl.GetProgramBuildInfo != Success");
                Console.WriteLine(Cl.GetProgramBuildInfo(program, _device, ProgramBuildInfo.Log, out error));
                return;
            }

            //Create the required kernel (entry function)
            Kernel kernel = Cl.CreateKernel(program, "dot", out error);
            CheckErr(error, "Cl.CreateKernel");

            //int M = 5;
            //int N = 10;
            //int K = 15;

            //double[,] A = new double[M, K];
            //double[,] B = new double[K, N];
            //double[,] C = new double[M, N];

            //var queue = Cl.CreateCommandQueue(_context, _device, (CommandQueueProperties)0, out error);

            //var buffA = Cl.CreateBuffer(_context, MemFlags.ReadOnly, M * K * Marshal.SizeOf(A), out error);
            //var buffB = Cl.CreateBuffer(_context, MemFlags.ReadOnly, K * N * Marshal.SizeOf(A), out error);
            //var buffB_TR = Cl.CreateBuffer(_context, MemFlags.ReadOnly, N * K * Marshal.SizeOf(A), out error);
            //var buffC = Cl.CreateBuffer(_context, MemFlags.ReadOnly, M * N * Marshal.SizeOf(A), out error);

            ////Cl.EnqueueWriteBuffer(queue, buffA, Bool.True, 0, M * K * Marshal.SizeOf(A), A, 0, null, null);

            //Cl.SetKernelArg(kernel, 0, sizeof(int), (void*)&M);
            //Cl.SetKernelArg(kernel, 1, sizeof(int), (void*)&N);
            //Cl.SetKernelArg(kernel, 2, sizeof(int), (void*)&K);
            //Cl.SetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&A);
            //Cl.SetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&B);
            //Cl.SetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&C);
            //CheckErr(error, "Cl.SetKernelArg");
        }
    }
}
