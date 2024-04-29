//*****************************************************************************************
//*                                                                                       *
//* This is an auto-generated file by Microsoft ML.NET CLI (Command-Line Interface) tool. *
//*                                                                                       *
//*****************************************************************************************

using System;
using System.IO;
using System.Linq;
using SampleClassification.Model;

namespace TitanicApp
{
    class Program
    {
        static void Main(string[] args)
        {

            ModelBuilder.CreateModel();
            
            // Create single instance of sample data from first line of dataset for model input
            ModelInput sampleData = new ModelInput()
            {
                Col0 = 0F,
                PassengerId = 1F,
                Sex = 1F,
                Age = 0.275F,
                Fare = 0.014151057F,
                Pclass_1 = 0F,
                Pclass_2 = 0F,
                Pclass_3 = 1F,
                Family_size = 0.1F,
                Title_1 = 1F,
                Title_2 = 0F,
                Title_3 = 0F,
                Title_4 = 0F,
                Emb_1 = 0F,
                Emb_2 = 0F,
                Emb_3 = 1F,
            };

            // Make a single prediction on the sample data and print results
            var predictionResult = ConsumeModel.Predict(sampleData);

            Console.WriteLine("Using model to make single prediction " +
                              "-- Comparing actual Survived with predicted Survived from sample data...\n\n");
            Console.WriteLine($"Col0: {sampleData.Col0}");
            Console.WriteLine($"PassengerId: {sampleData.PassengerId}");
            Console.WriteLine($"Sex: {sampleData.Sex}");
            Console.WriteLine($"Age: {sampleData.Age}");
            Console.WriteLine($"Fare: {sampleData.Fare}");
            Console.WriteLine($"Pclass_1: {sampleData.Pclass_1}");
            Console.WriteLine($"Pclass_2: {sampleData.Pclass_2}");
            Console.WriteLine($"Pclass_3: {sampleData.Pclass_3}");
            Console.WriteLine($"Family_size: {sampleData.Family_size}");
            Console.WriteLine($"Title_1: {sampleData.Title_1}");
            Console.WriteLine($"Title_2: {sampleData.Title_2}");
            Console.WriteLine($"Title_3: {sampleData.Title_3}");
            Console.WriteLine($"Title_4: {sampleData.Title_4}");
            Console.WriteLine($"Emb_1: {sampleData.Emb_1}");
            Console.WriteLine($"Emb_2: {sampleData.Emb_2}");
            Console.WriteLine($"Emb_3: {sampleData.Emb_3}");
            Console.WriteLine($"\n\nPredicted Survived value {predictionResult.Prediction} " +
                              $"\nPredicted Survived scores: [{String.Join(",", predictionResult.Score)}]\n\n");
            Console.WriteLine("=============== End of process, hit any key to finish ===============");

            var lines = File.ReadAllLines("../../../data/titanic-test-data.csv").Skip(1).ToArray();
            var counter = 0F;
            var correct = 0F;

            foreach (var line in lines)
            {
                var values = line.Split(',');
                // Get the actual result from the file
                var result = float.Parse(values[2]);

                // Create an instance of sample data for each line in the file
                sampleData = new ModelInput()
                {
                    Col0 = float.Parse(values[0]),
                    PassengerId = float.Parse(values[1]),
                    Sex = float.Parse(values[3]),
                    Age = float.Parse(values[4]),
                    Fare = float.Parse(values[5]),
                    Pclass_1 = float.Parse(values[6]),
                    Pclass_2 = float.Parse(values[7]),
                    Pclass_3 = float.Parse(values[8]),
                    Family_size = float.Parse(values[9]),
                    Title_1 = float.Parse(values[10]),
                    Title_2 = float.Parse(values[11]),
                    Title_3 = float.Parse(values[12]),
                    Title_4 = float.Parse(values[13]),
                    Emb_1 = float.Parse(values[14]),
                    Emb_2 = float.Parse(values[15]),
                    Emb_3 = float.Parse(values[16]),
                };

                // Make a single prediction on the sample data and print results
                predictionResult = ConsumeModel.Predict(sampleData);

                Console.WriteLine(
                    "Using model to make single prediction -- Comparing actual Survived with predicted Survived from sample data...\n\n");

                Console.WriteLine(
                    $"\n\nPredicted Survived value {predictionResult.Prediction}  Actual result {result}" +
                    $"\nPredicted Survived scores: [{String.Join(",", predictionResult.Score)}]\n\n");

                correct += result == float.Parse(predictionResult.Prediction) ? 1 : 0;
                counter++;
            }
            if(counter > 0)
                Console.WriteLine($"Predicted {counter} correct {correct} Percentage {correct/counter * 100}%");
        
        
            Console.ReadKey();
            
            
            
        }
    }
}
