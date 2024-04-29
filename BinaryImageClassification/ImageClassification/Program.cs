using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using Microsoft.ML;


namespace ImageClassification
{
    
    class ImageData
    {
        public string ImagePath { get; set; }

        public string Label { get; set; }
    }

    class ModelInput
    {
        public byte[] Image { get; set; }
        
        public UInt32 LabelAsKey { get; set; }

        public string ImagePath { get; set; }

        public string Label { get; set; }
    }

    class ModelOutput
    {
        public string ImagePath { get; set; }

        public string Label { get; set; }

        public string PredictedLabel { get; set; }
    }
    class Program
    {
        static void Main(string[] args)
        {
            
            var projectDirectory = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../"));
            var assetsRelativePath = Path.Combine(projectDirectory, "assets");

            var mlContext = new MLContext();

            var images = LoadImagesFromDirectory(folder: assetsRelativePath, useFolderNameAsLabel: true);

            // Load up the image data
            var imageData = mlContext.Data.LoadFromEnumerable(images);

            // Shuffle the data so we don't always train against the same data points
            var shuffledData = mlContext.Data.ShuffleRows(imageData);

            // Create our preprocessing pipeline  
            var preprocessingPipeline = mlContext.Transforms.Conversion.MapValueToKey(
                    inputColumnName: "Label",
                    outputColumnName: "LabelAsKey")
                .Append(mlContext.Transforms.LoadRawImageBytes(
                    outputColumnName: "Image",
                    imageFolder: assetsRelativePath,
                    inputColumnName: "ImagePath"));

            var preProcessedData = preprocessingPipeline
                                .Fit(shuffledData)
                                .Transform(shuffledData);

            // Split the data into train and test sets
            var trainSplit = mlContext.Data.TrainTestSplit(data: preProcessedData, testFraction: 0.2);
            var validationTestSplit = mlContext.Data.TrainTestSplit(trainSplit.TestSet);

            var trainSet = trainSplit.TrainSet;
            var testSet = validationTestSplit.TestSet;
            
            
            var pipeline = mlContext.MulticlassClassification.Trainers
                .ImageClassification(featureColumnName: "Image",
                    labelColumnName: "LabelAsKey",
                    validationSet: testSet)
                .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: "PredictedLabel",
                    inputColumnName: "PredictedLabel"));

            var trainedModel = pipeline.Fit(trainSet);

            ClassifySingleImage(mlContext, testSet, trainedModel);

            ClassifyImages(mlContext, testSet, trainedModel);

            Console.ReadKey();
        }

        private static void ClassifySingleImage(MLContext mlContext, IDataView data, ITransformer trainedModel)
        {
            var predictionEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(trainedModel);

            var image = mlContext.Data.CreateEnumerable<ModelInput>(data,reuseRowObject:true).First();

            var prediction = predictionEngine.Predict(image);

            Console.WriteLine("Classifying single image");
            OutputPrediction(prediction);
        }

        private static void ClassifyImages(MLContext mlContext, IDataView data, ITransformer trainedModel)
        {
            var predictionData = trainedModel.Transform(data);

            var predictions = mlContext.Data.CreateEnumerable<ModelOutput>(predictionData, reuseRowObject: true).Take(10);

            Console.WriteLine("Classifying multiple images");
            foreach (var prediction in predictions)
            {
                OutputPrediction(prediction);
            }
        }

        private static void OutputPrediction(ModelOutput prediction)
        {
            var imageName = Path.GetFileName(prediction.ImagePath);
            Console.WriteLine($"Image: {imageName} | Actual Value: {prediction.Label} | Predicted Value: {prediction.PredictedLabel}");
        }

        private static IEnumerable<ImageData> LoadImagesFromDirectory(string folder, bool useFolderNameAsLabel = true)
        {
            var files = Directory.GetFiles(folder, "*",
                searchOption: SearchOption.AllDirectories);

            foreach (var file in files)
            {
                if ((Path.GetExtension(file) != ".jpg") && (Path.GetExtension(file) != ".png"))
                    continue;

                var label = Path.GetFileName(file);

                if (useFolderNameAsLabel)
                    label = Directory.GetParent(file)?.Name;
                else
                {
                    for (var index = 0; index < label.Length; index++)
                    {
                        if (!char.IsLetter(label[index]))
                        {
                            label = label[..index];
                            break;
                        }
                    }
                }

                yield return new ImageData()
                {
                    ImagePath = file,
                    Label = label
                };
            }
        }
    }
}