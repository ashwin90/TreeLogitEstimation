data = readtable('raw1000/sushi.xlsx');
data = table2array(data);
data = data+1;

n=1000;
p=0.7;
nobuy = ones(n,1);

for nb = 1:400
     offerset = binornd(1,p*ones(n,100));
     offerset = [nobuy,offerset];
     sales= zeros(n,101);
     
     
     for i = 1:n
         %k = randperm(5000);
         preference_data = data;
         for j = 1:5000
             buy_flag = 0;
             for k = 1:10
                 index = preference_data(j,k);
                 if offerset(i, index+1) == 1
                     sales(i, index+1) = sales(i, index+1)+1;
                     buy_flag = 1;
                     break;
                 end
             end

             if buy_flag == 0
                 sales(i, 1)=sales(i, 1)+1;
             end

         end
     end
     
     k = randperm(n);
     n_train = n/10*7;
       
     training_offerset = offerset(k(1:n_train),:);
     testing_offerset = offerset(k((n_train+1):n),:);
     
     training_sales = sales(k(1:n_train),:);
     testing_sales = sales(k(n_train+1:n),:);
     
     
     %   training_data = struct2cell(data.transactions.in_sample_transactions);
     %   testing_data = struct2cell(data.transactions.out_of_sample_transactions);
     

   save(['data_0405_07/',num2str(nb),'.mat'],'training_offerset','training_sales','testing_sales','testing_offerset');
   fprintf('%f\n',nb);
   
end