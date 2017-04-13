bool descriptorDistance(std::vector<float> first,std::vector<float> second){

    vector<double> diffVec;
    double diffThreshold = 0.015;
    
    float elementDiff;
    double diffSquare, elementSum, diff, finalDiff;


    for(int i = 0; i < first.size(); i++){
        elementDiff = first.at(i) - second.at(i);
        diffSquare = pow(elementDiff,2);
        elementSum = first.at(i) + second.at(i);
        diff = diffSquare/elementDiff;
        diff = 2 * diff;
        finalDiff = diff / first.size();
        diffVec.push_back(finalDiff);

    }

    for(int i = 0; i < diffVec.size(); i++){
        if (diffVec.at(i) > diffThreshold){
             //cout << diffVec.at(i) << endl;
        }
       
        if (diffVec.at(i) < diffThreshold)
            return true;
    }
    return false;
}