function DemoNestedLoop()
    % dispaly documentation
    doc MultiWaitBar
    
    % instantiate the MultiWaitBar as a 3-by-1 grid
    mwb = MultiWaitBar(3, 1, 'Nested loop demo...', 'g');
    
    % initialize wait bars
    loopName = {'Outer loop progress...', 'Middle loop progress...', 'Inner loop progress...'};
    for ix = 1:3 % initialize waitbars
        mwb.Update(ix, 1, 0, loopName{ix});
    end
    
    % nested loops with wait bar updates
    for ix1 = 1:3
        for ix2 = 1:5
            for ix3 = 1:100
                pause(.01);
                mwb.Update(3, 1, ix3 / 100);
            end
            mwb.Update(2, 1, ix2 / 5);
        end
        mwb.Update(1, 1, ix1 / 3);
    end
    
    % close the waitbar figure
    mwb.Close();
end