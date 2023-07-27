clear

%64*64
qmat_64=zeros(8,64,64);
for i=1:8
    qmat_64(i,:,:)=generate_GaussH(64, 16+i*4);
end

save 'qmats.mat'

%Gaussian lowpass filter
function H=generate_GaussH(H_size, D0)
    H=zeros(H_size);
    for u=1:H_size
        for v=1:H_size
            D2=u*u+v*v;
            H(u,v)=exp(-D2/(2*D0^2));
        end
    end
    H=mapminmax(H(:),0,1);
    H=reshape(H,H_size,H_size);
end
