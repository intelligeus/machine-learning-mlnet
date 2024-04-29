using Microsoft.ML;
using MultiImageClassification.Common;
using MultiImageClassification.Model;

namespace MultiImageClassification.Tensorflow
{
    public class TensorflowScorer(string tagsLocation, string imagesFolder, string modelLocation, string labelsLocation)
    {
        private readonly string _imagesFolder = imagesFolder;

        private readonly MLContext _mlContext = new();
        //private static string _imageReal = nameof(_imageReal);

        private struct ImageSettings
        {
            public const int ImageHeight = 224;
            public const int ImageWidth = 224;
            public const float Mean = 117;
            public const bool ChannelsLast = true;
        }
        
        public struct InceptionSettings
        {
            // input tensor name
            public const string InputTensorName = "input";

            // output tensor name
            public const string OutputTensorName = "softmax2";
        }
        
        public void Score()
        {
            var model = LoadModel(tagsLocation, _imagesFolder, modelLocation);

            var predictions = PredictImageWithEngine(tagsLocation, _imagesFolder, labelsLocation, model).ToArray();

        }

        private PredictionEngine<Image, ImagePrediction> LoadModel(string dataLocation, string imagesFolder, string modelLocation)
        {
            ConsoleHelper.ConsoleWriteHeader("Setup and read the pre-trained model");
            Console.WriteLine($"Model location: {modelLocation}");
            Console.WriteLine($"Images folder: {imagesFolder}");
            Console.WriteLine($"Training file: {dataLocation}");
            Console.WriteLine($"Default parameters: image size=({ImageSettings.ImageWidth},{ImageSettings.ImageHeight}), image mean: {ImageSettings.Mean}");

            var data = _mlContext.Data.LoadFromTextFile<Image>(dataLocation, hasHeader: true);

            var pipeline = _mlContext.Transforms.LoadImages(outputColumnName: "input", imageFolder: imagesFolder, inputColumnName: nameof(Image.ImagePath))
                            .Append(_mlContext.Transforms.ResizeImages(outputColumnName: "input", imageWidth: ImageSettings.ImageWidth, imageHeight: ImageSettings.ImageHeight, inputColumnName: "input"))
                            .Append(_mlContext.Transforms.ExtractPixels(outputColumnName: "input", interleavePixelColors: ImageSettings.ChannelsLast, offsetImage: ImageSettings.Mean))
                            .Append(_mlContext.Model.LoadTensorFlowModel(modelLocation).
                            ScoreTensorFlowModel(outputColumnNames: ["softmax2"],
                                                inputColumnNames: ["input"], addBatchDimensionInput:true));
                        
            ITransformer model = pipeline.Fit(data);

            var predictionEngine = _mlContext.Model.CreatePredictionEngine<Image, ImagePrediction>(model);

            return predictionEngine;
        }

        private IEnumerable<Image> PredictImageWithEngine(string testLocation, 
                                                                  string imagesFolder, 
                                                                  string labelsLocation, 
                                                                  PredictionEngine<Image, ImagePrediction> model)
        {
            ConsoleHelper.ConsoleWriteHeader("Classify images");
            Console.WriteLine($"Images folder: {imagesFolder}");
            Console.WriteLine($"Training file: {testLocation}");
            Console.WriteLine($"Labels file: {labelsLocation}");

            var labels = ReadLabels(labelsLocation);

            var testData = Image.ReadFromCsv(testLocation, imagesFolder);

            foreach (var sample in testData)
            {
                var probs = model.Predict(sample).PredictedLabels;
                var imageData = new ImageProbability()
                {
                    ImagePath = sample.ImagePath,
                    Label = sample.Label
                };
                (imageData.PredictedAs, imageData.Probability) = GetBestLabel(labels, probs);
                imageData.ConsoleWrite();
                yield return imageData;
            }
        }
        
        private (string,float) GetBestLabel(string[] labels, float[] probs)
        {
            var max = probs.Max();
            var index = probs.AsSpan().IndexOf(max);
            return (labels[index],max);
        }

        private string[] ReadLabels(string labelsLocation)
        {
            return File.ReadAllLines(labelsLocation);
        }
    }
}
