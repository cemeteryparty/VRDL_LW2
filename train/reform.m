%load digitStruct.mat

for i = 1:length(digitStruct)
    bbox = zeros(length(digitStruct(i).bbox), 5);
    for j = 1:length(digitStruct(i).bbox)
        bbox(j,1)=digitStruct(i).bbox(j).height;
        bbox(j,2)=digitStruct(i).bbox(j).left;
        bbox(j,3)=digitStruct(i).bbox(j).top;
        bbox(j,4)=digitStruct(i).bbox(j).width;
        bbox(j,5)=digitStruct(i).bbox(j).label;
    end
    save(['mat/', digitStruct(i).name, '_bbox.mat'], 'bbox');
end