//*****************************************************************************************
//*                                                                                       *
//* This is an auto-generated file by Microsoft ML.NET CLI (Command-Line Interface) tool. *
//*                                                                                       *
//*****************************************************************************************

using Microsoft.ML.Data;

namespace SampleClassification.Model
{
    public class ModelInput
    {
        [ColumnName("col0"), LoadColumn(0)]
        public float Col0 { get; set; }


        [ColumnName("PassengerId"), LoadColumn(1)]
        public float PassengerId { get; set; }


        [ColumnName("Survived"), LoadColumn(2)]
        public string Survived { get; set; }


        [ColumnName("Sex"), LoadColumn(3)]
        public float Sex { get; set; }


        [ColumnName("Age"), LoadColumn(4)]
        public float Age { get; set; }


        [ColumnName("Fare"), LoadColumn(5)]
        public float Fare { get; set; }


        [ColumnName("Pclass_1"), LoadColumn(6)]
        public float Pclass_1 { get; set; }


        [ColumnName("Pclass_2"), LoadColumn(7)]
        public float Pclass_2 { get; set; }


        [ColumnName("Pclass_3"), LoadColumn(8)]
        public float Pclass_3 { get; set; }


        [ColumnName("Family_size"), LoadColumn(9)]
        public float Family_size { get; set; }


        [ColumnName("Title_1"), LoadColumn(10)]
        public float Title_1 { get; set; }


        [ColumnName("Title_2"), LoadColumn(11)]
        public float Title_2 { get; set; }


        [ColumnName("Title_3"), LoadColumn(12)]
        public float Title_3 { get; set; }


        [ColumnName("Title_4"), LoadColumn(13)]
        public float Title_4 { get; set; }


        [ColumnName("Emb_1"), LoadColumn(14)]
        public float Emb_1 { get; set; }


        [ColumnName("Emb_2"), LoadColumn(15)]
        public float Emb_2 { get; set; }


        [ColumnName("Emb_3"), LoadColumn(16)]
        public float Emb_3 { get; set; }


    }
}
