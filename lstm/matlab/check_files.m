function flag = check_files(family_i, family_f, tS, load_dir)
%returns 0 if all the files exist, -1 if not
    flag = 0;
    for i = family_i : family_f
        id = fopen([load_dir int2str(i) '/tS_' int2str(tS)], 'r');
        if id < 0
            flag = -1;
            break;
        else
            fclose(id);
        end
    end
end