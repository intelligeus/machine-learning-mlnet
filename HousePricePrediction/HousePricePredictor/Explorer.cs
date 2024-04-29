using Microsoft.Data.Analysis;
namespace HousePricePredictor
{
    
    public class Explorer
    {
        public void ExploreData()
        {
            var frame = DataFrame.LoadCsv("../Data./train.csv");
            
        }
    }
}