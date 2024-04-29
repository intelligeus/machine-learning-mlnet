﻿using MultiImageClassification.Model;

namespace MultiImageClassification.Common
{
    public static class ConsoleHelper
    {
        public static void ConsoleWriteHeader(params string[] lines)
        {
            var defaultColor = Console.ForegroundColor;
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine(" ");
            foreach (var line in lines)
            {
                Console.WriteLine(line);
            }
            var maxLength = lines.Select(x => x.Length).Max();
            Console.WriteLine(new String('#', maxLength));
            Console.ForegroundColor = defaultColor;
        }

        public static void ConsolePressAnyKey()
        {
            var defaultColor = Console.ForegroundColor;
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine(" ");
            Console.WriteLine("Press any key to finish.");
            Console.ForegroundColor = defaultColor;
            Console.ReadKey();
        }

        public static void ConsoleWriteException(params string[] lines)
        {
            var defaultColor = Console.ForegroundColor;
            const string exceptionTitle = "EXCEPTION";

            Console.WriteLine(" ");
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine(exceptionTitle);
            Console.WriteLine(new String('#', exceptionTitle.Length));
            Console.ForegroundColor = defaultColor;
            foreach (var line in lines)
            {
                Console.WriteLine(line);
            }
        }

        public static void ConsoleWrite(this ImageProbability self)
        {
            var defaultForeground = Console.ForegroundColor;
            var labelColor = ConsoleColor.Magenta;
            var probColor = ConsoleColor.Blue;
            var exactLabel = ConsoleColor.Green;
            var failLabel = ConsoleColor.Red;

            Console.Write("ImagePath: ");
            Console.ForegroundColor = labelColor;
            Console.Write($"{Path.GetFileName(self.ImagePath)}");
            Console.ForegroundColor = defaultForeground;
            Console.Write(" labeled as ");
            Console.ForegroundColor = labelColor;
            Console.Write(self.Label);
            Console.ForegroundColor = defaultForeground;
            Console.Write(" predicted as ");
            if (self.Label.Equals(self.PredictedAs))
            {
                Console.ForegroundColor = exactLabel;
                Console.Write($"{self.PredictedAs}");
            }
            else
            {
                Console.ForegroundColor = failLabel;
                Console.Write($"{self.PredictedAs}");
            }
            Console.ForegroundColor = defaultForeground;
            Console.Write(" with probability ");
            Console.ForegroundColor = probColor;
            Console.Write(self.Probability);           
            Console.ForegroundColor = defaultForeground;
            Console.WriteLine("");
        }

    }

}
