using System;
using System.Globalization;
using System.IO;
using CsvHelper;
using CsvHelper.Configuration;
using Microsoft.Data.Analysis;

namespace HousePricePredictor
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            var config = new CsvConfiguration(CultureInfo.InvariantCulture)
            {
                PrepareHeaderForMatch = args => args.Header.ToLower(),
            };
            using var reader = new StreamReader("/home/marty/Repos/machine-learning-examples/HousePricePrediction/Data/train.csv");
            using (var csv = new CsvReader(reader, config))
            {
                var records = csv.GetRecords<string>();
                Console.WriteLine(records);
            }
        }
    }
}