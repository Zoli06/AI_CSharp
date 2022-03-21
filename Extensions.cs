using System;

namespace AI
{
    public static class BooleanExtensions
    {
        public static int ToInt(this bool value)
        {
            return value ? 1 : 0;
        }
    }

    static class RandomExtensions
    {
        public static void Shuffle<T>(this Random rng, T[] array)
        {
            int n = array.Length;
            while (n > 1)
            {
                int k = rng.Next(n--);
                T temp = array[n];
                array[n] = array[k];
                array[k] = temp;
            }
        }

        public static void Shuffle<T>(this Random rng, T[][] array)
        {
            int n = array.GetLength(0);
            while (n > 1)
            {
                int k = rng.Next(n--);
                T[] temp = array[n];
                array[n] = array[k];
                array[k] = temp;
            }
        }
    }
}
