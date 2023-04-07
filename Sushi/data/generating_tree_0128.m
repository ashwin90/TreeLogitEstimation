
T = readtable('sushi.xlsx');   


node1 = T(T.Style== 1, :);
node2 = T(T.Style== 0, :);

node3 = node1(node1.MajorGroup== 0, :);
node4 = node1(node1.MajorGroup== 1, :);
node5 = node2(node2.MajorGroup== 0, :);
node6 = node2(node2.MajorGroup== 1, :);

node7 = node3(node3.MinorGroup== 0, :);
node8 = node3(node3.MinorGroup== 1, :);
node9 = node3(node3.MinorGroup== 2, :);
node10 = node3(node3.MinorGroup== 3, :);
node11 = node3(node3.MinorGroup== 4, :);
node12 = node3(node3.MinorGroup== 5, :);
node13 = node3(node3.MinorGroup== 6, :);
node14 = node3(node3.MinorGroup== 7, :);
node15 = node3(node3.MinorGroup== 8, :);
node16 = node4(node4.MinorGroup== 9, :);
node17 = node4(node4.MinorGroup== 10, :);
node18 = node4(node4.MinorGroup== 11, :);
node19 = node5(node5.MinorGroup== 1, :);
node20 = node5(node5.MinorGroup== 3, :);
node21 = node5(node5.MinorGroup== 4, :);
node22 = node5(node5.MinorGroup== 7, :);
node23 = node6(node6.MinorGroup== 11, :);

node24 = node7(node7.HeavinessInTaste <= 2,:).ID;
node25 = node7(node7.HeavinessInTaste > 2,:).ID;
node26 = node8(node8.HeavinessInTaste <= 2,:).ID;
node27 = node9(node9.HeavinessInTaste <= 2,:).ID;
node28 = node9(node9.HeavinessInTaste > 2,:).ID;
node29 = node10(node10.HeavinessInTaste <= 2,:).ID;
% 
node30 = node11(node11.HeavinessInTaste <= 2,:).ID;
node31 = node11(node11.HeavinessInTaste > 2,:).ID;
node32 = node12(node12.HeavinessInTaste <= 2,:).ID;
node33 = node12(node12.HeavinessInTaste > 2,:).ID;
node34 = node13(node13.HeavinessInTaste <= 2,:).ID;
node35 = node13(node13.HeavinessInTaste > 2,:).ID;
node36 = node14(node14.HeavinessInTaste <= 2,:).ID;
node37 = node14(node14.HeavinessInTaste > 2,:).ID;
node38 = node15(node15.HeavinessInTaste <= 2,:).ID;
node39 = node15(node15.HeavinessInTaste > 2,:).ID;
node40 = node16(node16.HeavinessInTaste > 2,:).ID;
node41 = node17(node17.HeavinessInTaste <= 2,:).ID;
node42 = node17(node17.HeavinessInTaste > 2,:).ID;
node43 = node18(node18.HeavinessInTaste <= 2,:).ID;
node44 = node18(node18.HeavinessInTaste > 2,:).ID;
node45 = node19(node19.HeavinessInTaste <= 2,:).ID;
node46 = node19(node19.HeavinessInTaste > 2,:).ID;
node47 = node20.ID;
node48 = node21.ID;
node49 = node22.ID;
node50 = node23(node23.HeavinessInTaste <= 2,:).ID;
node51 = node23(node23.HeavinessInTaste > 2,:).ID;


adj_mats = {ones(3, 1)};
    level2 = zeros(5,3);
    level2(1,1)=1;
    level2(2,2)=1;
    level2(3,2)=1;
    level2(4,3)=1;
    level2(5,3)=1;
    adj_mats{2} = level2;
    
    level3 = zeros(18,5);
    level3(1,1)=1;
    for i = 2:10
        level3(i,2)=1;
    end
    level3(11,3)=1;
    level3(12,3)=1;
    level3(13,3)=1;
    level3(14,4)=1;
    level3(15,4)=1;
    level3(16,4)=1;
    level3(17,4)=1;
    level3(18,5)=1;
    adj_mats{3} = level3;
    
    level4 = zeros(29,18);
    level4(1,1)=1;
    level4(2,2)=1;
    level4(3,2)=1;
    level4(4,3)=1;
    level4(5,4)=1;
    level4(6,4)=1;
    level4(7,5)=1;
    for i = 8:17
        level4(i,floor(i/2)+2)=1;
    end
    level4(18,11)=1;
    for i = 19:24
        level4(i,floor((i+5)/2))=1;
    end
    level4(25,15)=1;
    level4(26,16)=1;
    level4(27,17)=1;
    level4(28,18)=1;
    level4(29,18)=1;
    adj_mats{4} = level4;
    
    
    level5 = zeros(101,29);
    level5(1,1) =1;
    item = 2;
    index_order = [];
    for i = 2:29
        nb = eval(['length(node',num2str(i+22),')']);
        index_order = [index_order , eval(['node',num2str(i+22)])'];
        for j = item:(item+nb-1)
            level5(j,i)=1;
        end
        item = item+nb;
    end

    adj_mats{5} = level5;
    adj_mats = fliplr(adj_mats);


index_order = [1,2+index_order];


save(['final_tree_0128/node.mat'],'index_order','adj_mats');

