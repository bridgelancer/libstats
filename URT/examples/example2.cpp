//=================================================================================================
//                    Copyright (C) 2016 Olivier Mallet - All Rights Reserved                      
//=================================================================================================

#include "../include/URT.hpp"
#include "../include/CsvManager.hpp"

int main()
{

   urt::Vector<double> data;
   // Read from Training.csv
   std::string filename = "Training.csv";
   data.load(filename);

   for (int i = 0; i< 23; i++){
     urt::ADF<double> test(data, i, "ct");
     test.show();  
   }
   // outputting test results
   
}
