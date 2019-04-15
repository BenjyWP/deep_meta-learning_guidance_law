cell_run = cell(100,1);
cell_los_MD = cell(100,1);
cell_MD = cell(100,1);
los_basis = 'last_los_MD';
run_basis = 'last_run';
md_basis = 'last_MD';

for i =1:1:100
   name_los = sprintf('%s%d%s',los_basis,i,'.npy'); 
   name_run = sprintf('%s%d%s',run_basis,i,'.npy');
   name_MD = sprintf('%s%d%s',md_basis,i,'.npy');
   cell_los_MD{i} = readNPY(name_los);
   cell_run{i} = readNPY(name_run);
   cell_MD{i} = readNPY(name_MD);
   
end

