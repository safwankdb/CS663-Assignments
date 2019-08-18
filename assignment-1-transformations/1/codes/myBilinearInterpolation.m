function [ output_image ]  = myBilinearInterpolation(input,zoom_row,zoom_col)

[im , map] =imread(input);

[row, col] = size(im); 
%zooming factor
zr=zoom_row*row - (zoom_row - 1);
zc=zoom_col*col - (zoom_col - 1);

for i=1:zr
    i_index = i -1;
    x=i_index/zoom_row;
    
    x1=floor(x);
    x2=ceil(x);

    x1 = x1 +1;
    x2 = x2 +1;
    xint=rem(x,1);
    for j=1:zc
        j_index = j -1;
        y=j_index/zoom_col;
        
        y1=floor(y);
        y2=ceil(y);

        yint=rem(y,1);
        y1 = y1 +1;
        y2 = y2 +1;
        
        if (x2 <= row & x1 <= row & y1 <= col & y2 <= col)

            BL=im(x1,y1);
            TL=im(x1,y2);
            BR=im(x2,y1);
            TR=im(x2,y2);
        
            output_image(i,j) = BR*xint*(1-yint) + BL*(1-xint)*(1-yint) + TR*xint*yint + TL*(1-xint)*yint ;
        end
    end
end