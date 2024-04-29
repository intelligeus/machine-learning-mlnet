using Microsoft.ML.Data;

namespace MultiImageClassification.Model
{
    // Class to manage Image paths and Labels
    public class Image
    {
        [LoadColumn(0)]
        public string ImagePath;

        [LoadColumn(1)]
        public string Label;

        public static IEnumerable<Image> ReadFromCsv(string file, string folder)
        {
            return File.ReadAllLines(file)
             .Select(x => x.Split('\t'))
             .Select(x => new Image { ImagePath = Path.Combine(folder, x[0]), Label = x[1] } );
        }
    }

    // Class to contain the predictions the model generates
    public class ImageProbability : Image
    {
        public string PredictedAs;
        public float Probability { get; set; }
    }
}
