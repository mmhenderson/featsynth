% OBJECT CATEGORY JUDGMENT TASK
 
try 
    
    echo off  
    clear 
    close all  hidden
       
    % p is the structure that will hold everything we want to save
    p.scanner_laptop = 0;
   

    expdir = pwd;
    filesepinds = find(expdir==filesep);
    root = expdir(1:filesepinds(end)-1);
    
    % set up paths for saving
    datadir_local = fullfile(root, 'fmri_behav_data');
    
    %% Collect information about the subject, the date, etc.
    
    % make a dialog box to enter things
    prompt = {'Debug mode?','Subject Initials','Subject Number',...
        'Session (1-3)','Run Number (1-16)'};
    dlgtitle = 'Enter Run Parameters';
    dims = [1 35];
    definput = {'1','XX','99','1','1'};
    answer = inputdlg(prompt,dlgtitle,dims,definput);
    p.debug = str2double(answer{1});
    p.sub_init = answer{2};  
    p.sub_num = str2double(answer{3});
    p.sub_num_str = sprintf('%02d',str2double(answer{3}));
    p.session = str2double(answer{4});
    p.run_num = str2double(answer{5});
  
    % check values of these entries
    if ~ismember(p.session,1:2);  error('session must be 1,2,3');  end
    if ~ismember(p.run_num, 1:16); error('run must be 1-16'); end
   
    rng('default')
    p.rndseed = sum(100*clock); 
    rng(p.rndseed);
     
    % where will i look for images? 
    if p.scanner_laptop
        p.image_dir = ''; % todo: fix
    else
        p.image_dir = '/Users/margarethenderson/Dropbox/Apps/featsynth/images_comb64/';
    end

    %% initialize my data file
    % save a copy of the currently running script here (in text format) so
    % we can look at it later if issues arise
    p.script_text = fileread([mfilename('fullpath'),'.m']);
    
    % make sure my data dir exists
    if ~exist(datadir_local,'dir')
        mkdir(datadir_local)
    end
   
    p.date_str = datestr(now,'yymmdd'); %Collect todays date (in p.)
    p.time_str = datestr(now,'HHMM'); %Timestamp for saving out a uniquely named datafile (so you will never accidentally overwrite stuff)
         
    p.exp_name = 'maintask_fMRI_pilot1';
   
    % this one file will hold all runs for this session/day
    p.fnsave_local = fullfile(datadir_local, ...
        ['S', p.sub_num_str, '_' p.exp_name '_' p.date_str '.mat']);
   
    if exist(p.fnsave_local,'file')
        % load the existing file, will append to it
        load(p.fnsave_local, 'dat_all');
        % in this file is "dat_all", which lists info from prev runs
        if p.run_num~=length(dat_all)+1
             error('Check your run number, %d runs have been completed',length(dat_all));
        end
        for rr = 1:length(dat_all)
            % checking some stuff about previous data
            assert(dat_all(rr).p.session==p.session)
        end
    elseif p.run_num~=1            
        error('No data exists yet for this subject, check your run number')
    end
    
    p.rndseed = round(sum(100*clock));
    rng(p.rndseed);
    
    if p.debug
        p.num_trials = 5;
    else   
        p.num_trials = 40;
    end

    %% figure out what to do during this block.

    % all the trial info is stored in a .json file that i made in python
    expt_design_path = fullfile(root,'expt_design','fmri_pilot1');
    p.expt_design_file = fullfile(expt_design_path, 'trialseq.json');
    % load the json here, it has all the different random orders within it.
    trial_info_all = jsondecode(fileread(p.expt_design_file));

    % this file tells which random order to use for this subjects
    rnd_order_fn = fullfile(root, 'fmri_expt_code', 'random_orders.mat');
    load(rnd_order_fn);
    rand_order_num = rndorder(p.sub_num);

    % index into runs for whole experiment
    p.run_num_overall = p.run_num + (p.session-1)*16;

    % todo: trial info should include both sessions.
    
    % ti is everything we need to run the expt. 
    % [n_runs x n_trials]
    % where n_runs is across ALL SESSIONS
    ti = squeeze(trial_info_all(rand_order_num, :, :));
    
    % pulling out some fields that we need for this run
    p.miniblock_number = [ti(p.run_num_overall,:).miniblock_number_in_run];
    p.miniblock_task = {ti(p.run_num_overall,:).cue_level};

    p.n_trials_mini = sum(p.miniblock_number==1);
    p.n_miniblocks = length(unique(p.miniblock_number));

    p.correct_resp = [ti(p.run_num_overall,:).correct_resp]';
    p.image_names = {ti(p.run_num_overall,:).image_name};
    p.super_names = {ti(p.run_num_overall,:).super_name};
    p.basic_names = {ti(p.run_num_overall,:).basic_name};
    p.left_names = {ti(p.run_num_overall,:).left_name};
    p.right_names = {ti(p.run_num_overall,:).right_name};
    p.cue_level = {ti(p.run_num_overall,:).cue_level};

    % replacing any underscores in the names with spaces
    rep = @(x) strrep(x, '_', ' ');
    p.left_names = cellfun(rep, p.left_names, 'UniformOutput', false);
    p.right_names = cellfun(rep, p.right_names, 'UniformOutput', false);
    p.super_names = cellfun(rep, p.super_names, 'UniformOutput', false);
    p.basic_names = cellfun(rep, p.basic_names, 'UniformOutput', false);
    
    p.random_order_number = ti(p.run_num_overall,1).random_order_number;


    %% set up my screen 
    
    InitializeMatlabOpenGL;  
    PsychImaging('PrepareConfiguration');
    if p.scanner_laptop
        Screen('Preference', 'SkipSyncTests', 0);
    else
        % this is just for testing, when we skip the sync tests this can
        % mess up the timing. for real expt this should always be 0
        Screen('Preference', 'SkipSyncTests', 1);
    end
    AssertOpenGL; % bail if current version of PTB does not use
    PsychJavaTrouble;
    
    s=max(Screen('Screens'));
    p.black = BlackIndex(s);
    p.white = WhiteIndex(s);
    % Open a screen
    Screen('Preference','VBLTimestampingMode',-1);  % for the moment, must disable high-precision timer on Win apps
    multiSample=0;
    
    % set background color of screen
    p.back_color = 77;   % round(0.3*255)
    
    [w, p.srect]=Screen('OpenWindow', s, p.back_color,[],[],[],multiSample);
    HideCursor;
    
    disp(p.srect)
    % Enable alpha blending with proper blend-function. We need it
    % for drawing of smoothed points:
    Screen('BlendFunction', w, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    % test the refresh properties of the display
    p.fps=Screen('FrameRate',w);          % frames per second
    p.ifi=Screen('GetFlipInterval', w);   % inter-frame-time
    if p.fps==0                           % if fps does not register, then set the fps based on ifi
        p.fps=1/p.ifi;
    end
 
    if p.scanner_laptop
        % TODO: fix for scanner laptop
        p.refresh_rate = 60;
        % TODO: get these measurements for screen at bridge center
        p.vdist_cm = 47;
        p.screen_height_cm = 16;
    else
        % macbook pro for testing
        p.refresh_rate = 60;
        p.vdist_cm = 40;
        p.screen_height_cm = 29;  

    end
    % visual angle of the whole screen
    p.vis_angle = (2*atan2(p.screen_height_cm/2, p.vdist_cm))*(180/pi); 

    % make sure the refreshrate is ok
    if abs(p.fps-p.refresh_rate)>5
        Screen('CloseAll');
        disp('CHANGE YOUR REFRESH RATE')
        ListenChar(0);
        %clear all;
        return;
    end

    
    HideCursor; % Hide the mouse cursor
    % set the priority up way high to discourage interruptions
    Priority(MaxPriority(w));
    
    %get center of screen
    p.center_pix = [(p.srect(3) - p.srect(1))/2, (p.srect(4) - p.srect(2))/2];
    
    % set some general stim params
    p.fix_size_deg = .5;
    p.fix_color = [0.8,0.8,0.8]*255;
    p.stim_height_deg = 12 ;   
    p.text_color = p.white;
    p.instr_text_height_deg = 1.2;
    p.miniblock_text_height_deg = 2.0; 

    % this is where the text cues on each trial appear
    p.cue_text_width_deg = 5.0;
    p.cue_text_height_deg = 4.0  ; 
    p.cue_text_x_offset_deg = 3.0;
    p.cue_text_y_offset_deg = 0;

    
    % convert from degrees to pixel units
    % this automatically makes a bunch of new p. variables
    p = deg2pix(p);  
    p.fix_size_pix = ceil(p.fix_size_pix);

    %% Load the images 
      
    for ii=1:p.num_trials
        
        imfn = fullfile(p.image_dir, ti(p.run_num_overall, ii).image_name);
        if exist(imfn,'file')
            im=imread(imfn);
        else  
            error('image file %s not found!',imfn)
        end        
        
        allims(ii).name=imfn;
        allims(ii).imtext=Screen('MakeTexture',w,im);
        p.imfns{ii} = imfn;

    end

    % set up a frame to plot the image in
    p.stim_width_pix = p.stim_height_pix*size(im,2)/size(im,1);
    p.frame_pos=[p.center_pix(1)-p.stim_width_pix/2,...
        p.center_pix(2)-p.stim_height_pix/2,...
        p.center_pix(1)+p.stim_width_pix/2,...
        p.center_pix(2)+p.stim_height_pix/2];

    % making rectangles to plot my cue text in, left and right of fix
    center_l = [p.center_pix(1) - p.cue_text_x_offset_pix, ...
                p.center_pix(2) - p.cue_text_y_offset_pix];
    center_r = [p.center_pix(1) + p.cue_text_x_offset_pix, ...
                p.center_pix(2) - p.cue_text_y_offset_pix];

    p.text_rect_l = [center_l(1)-p.cue_text_width_pix/2,...
        center_l(2)-p.cue_text_height_pix/2,...
        center_l(1)+p.cue_text_width_pix/2,...
        center_l(2)+p.cue_text_height_pix/2];
    
    p.text_rect_r = [center_r(1)-p.cue_text_width_pix/2,...
        center_r(2)-p.cue_text_height_pix/2,...
        center_r(1)+p.cue_text_width_pix/2,...
        center_r(2)+p.cue_text_height_pix/2];
    
    %% keys
    KbName('UnifyKeyNames')

    %use number pad - change this for scanner 
    if p.scanner_laptop
        % TODO: figure out what the keys will be at bridge center
        p.keys=[KbName('b'),KbName('y')];
    else
        p.keys=[KbName('u'),KbName('i')];
    end
    
    p.escape = KbName('escape');
    p.space = KbName('space');
    p.start = KbName('t'); % TODO: check about triggers at bridge center
    
      
    %% gamma correction
    % we are skipping this for this experiment

    gammacorrect = false;
    OriginalCLUT = [];
   
    
    %% Allocate arrays to store trial info
    
    % response each trial
    p.response = nan(p.num_trials,1); 

    % response time - from onset of text cues
    p.resp_time_from_onset = nan(p.num_trials,1); 
    
    % recording the raw times of every button press
    % useful for debugging or if the subject presses wrong button
    p.resp_raw_all = [];

    % [stim onset, stim offset, text onset, text offset]
    p.stim_flips = nan(p.num_trials,4);  

    %% timing information
    
    p.stim_time_sec= 0.500;      
    p.delay_time_sec = 1.5;
    p.cue_time_sec = 2.0;

    p.miniblock_instr_sec = 2.5;
    p.miniblock_delay_sec = 1.0;
    
    if p.scanner_laptop
        % actual timing for scanner expt - long delay at start,
        % longer/jittered ITIs. 
        p.start_fix_sec = 13;
        p.end_fix_sec = 8;
        p.iti_range_sec = [1, 5];
    else
        % for testing - whole thing can be shorter.
        p.start_fix_sec = 1;
        p.end_fix_sec = 0;
        p.iti_range_sec = [2, 2];
    end 
    
    % uniformly distributed random itis
    itis = linspace(p.iti_range_sec(1),p.iti_range_sec(2),p.num_trials);
    p.iti_sec = itis(randperm(length(itis)))';
   
    
    %% START EXPERIMENT
    % Draw an instruction screen, wait for space press
    FlushEvents('keyDown');
    Screen(w,'TextFont','Arial');
    Screen(w,'TextSize', 22);
 
    % draw the first instructions screen
    instr_text = 'Prepare for task.';
    DrawFormattedText(w, instr_text, 'center', ...
        p.center_pix(2)-p.instr_text_height_pix, p.text_color);

    
    Screen('DrawDots', w, [0,0], p.fix_size_pix, p.fix_color, p.center_pix, 0); 
    Screen('DrawingFinished', w);
    Screen('Flip', w);               

    resp_in_set=0;
    % wait for a trigger or space bar press to start
    while resp_in_set==0
        [resp_raw, resp_in_set, ts] = check_for_resp([p.start,p.space],p.escape);
        if resp_in_set==-1; escape_response(OriginalCLUT); end       
    end

    % start keeping track of timing params here.
    % this is when experiment started.
    p.start_exp_time = GetSecs; 

    % this timer will mark whenever the most recent event happened.
    % use this to make sure events last as long as they are supposed to.
    time_last_evt = p.start_exp_time;

    % tracking time since the start of the experiment.
    % this is mostly just for a sanity check, at the end make sure this
    % number is what we expect.
    time_from_start = 0;

    KbReleaseWait();
    
    %% Fixation period before starting the stimuli 
    FlushEvents('keyDown');
    Screen('Flip', w);
    ListenChar(2)
    
    Screen('DrawDots', w, [0,0], p.fix_size_pix, p.fix_color, p.center_pix, 0); 
    Screen('DrawingFinished', w);
    Screen('Flip', w);

    % Update timing - track duration since the last event ended
    time_in_this_evt = GetSecs - time_last_evt;  
    while (time_in_this_evt < p.start_fix_sec) % loop until duration passes
        time_in_this_evt = (GetSecs - time_last_evt);       
        [resp_raw, resp_in_set, ts] = check_for_resp(p.keys,p.escape);
        % check for "escape" responses only here
        if resp_in_set==-1; escape_response(OriginalCLUT); end 
        if resp_raw~=0
            % response, trial number, timestamp in expt, raw timestamp
            p.resp_raw_all = [p.resp_raw_all; [resp_raw, 0, ts-p.start_exp_time, ts]];
        end
    end

    % updating my timers here
    time_last_evt = time_last_evt + p.start_fix_sec;
    time_from_start = time_from_start + p.start_fix_sec;
    
    %% start trial loop

    for tt=1:p.num_trials

        % decide if this is the start of a new mini-block
        if mod(tt, p.n_trials_mini)==1  

            %% Instructions for start of mini-block
            if strcmp(p.miniblock_task{tt}, 'super')
                instr_text = sprintf('Starting Coarse Task.');
            else
                instr_text = sprintf('Starting Fine Task\nCategory: "%s"', ...
                    p.super_names{tt});
            end
 
            DrawFormattedText(w, instr_text, 'center', ...
                p.center_pix(2)-p.miniblock_text_height_pix, p.text_color);
        
            
            Screen('DrawDots', w, [0,0], p.fix_size_pix, p.fix_color, p.center_pix, 0); 
            Screen('DrawingFinished', w);
            Screen('Flip', w);               

             % Update timing - track  duration since the last event ended
            time_in_this_evt = GetSecs - time_last_evt; 
            while (time_in_this_evt < p.miniblock_instr_sec) % loop until duration passes
                time_in_this_evt = (GetSecs - time_last_evt);       
                [resp_raw, resp_in_set, ts] = check_for_resp(p.keys,p.escape);
                % check for "escape"  responses only here
                if resp_in_set==-1; escape_response(OriginalCLUT); end 
                if resp_raw~=0
                    % response, trial number, timestamp in expt, raw timestamp
                    p.resp_raw_all = [p.resp_raw_all; [resp_raw, tt, ts-p.start_exp_time, ts]];
                end
            end
        
            % updating my timers here
            time_last_evt = time_last_evt + p.miniblock_instr_sec;
            time_from_start = time_from_start + p.miniblock_instr_sec;

            %% Pause before showing first stimulus in the mini-block
            
            Screen('DrawDots', w, [0,0], p.fix_size_pix, p.fix_color, p.center_pix, 0); 
            Screen('DrawingFinished', w);
            Screen('Flip', w);               

             % Update timing - track duration since the last event ended
            time_in_this_evt = GetSecs - time_last_evt; 
            while (time_in_this_evt < p.miniblock_delay_sec) % loop until duration passes
                time_in_this_evt = (GetSecs - time_last_evt);       
                [resp_raw, resp_in_set, ts] = check_for_resp(p.keys,p.escape);
                % check for "escape" responses only here
                if resp_in_set==-1; escape_response(OriginalCLUT); end 
                if resp_raw~=0
                    % response, trial number, timestamp in expt, raw timestamp
                    p.resp_raw_all = [p.resp_raw_all; [resp_raw, tt, ts-p.start_exp_time, ts]];
                end
            end
        
            % updating my timers here
            time_last_evt = time_last_evt + p.miniblock_delay_sec;
            time_from_start = time_from_start + p.miniblock_delay_sec;

        end

        %% Show target image   
        
        Screen('DrawTexture', w, allims(tt).imtext,[],p.frame_pos);
        Screen('DrawDots', w, [0,0], p.fix_size_pix, p.fix_color, p.center_pix, 0); 
        Screen('DrawingFinished', w);
        [~, onset] = Screen('Flip', w); 
        p.stim_flips(tt,1) = onset; % onset of image

        
        % Update timing - track duration since the last event ended
        time_in_this_evt = GetSecs - time_last_evt; 
        while (time_in_this_evt < p.stim_time_sec) % loop until duration passes
            time_in_this_evt = (GetSecs - time_last_evt);       
            [resp_raw, resp_in_set, ts] = check_for_resp(p.keys,p.escape);
            % check for "escape" responses only here
            if resp_in_set==-1; escape_response(OriginalCLUT); end 
            if resp_raw~=0
                % response, trial number, timestamp in expt, raw timestamp
                p.resp_raw_all = [p.resp_raw_all; [resp_raw, tt, ts-p.start_exp_time, ts]];
            end
        end
    
        % updating my timers here
        time_last_evt = time_last_evt + p.stim_time_sec;
        time_from_start = time_from_start + p.stim_time_sec;
        
        %% Delay period (fix only)

        Screen('DrawDots', w, [0,0], p.fix_size_pix, p.fix_color, p.center_pix, 0); 
        Screen('DrawingFinished', w);
        [~, onset] = Screen('Flip', w); 
        p.stim_flips(tt,2) = onset; % offset of image

        % Update timing - track duration since the last event ended
        time_in_this_evt = GetSecs - time_last_evt; 
        while (time_in_this_evt < p.delay_time_sec) % loop until duration passes
            time_in_this_evt = (GetSecs - time_last_evt);       
            [resp_raw, resp_in_set, ts] = check_for_resp(p.keys,p.escape);
            % check for "escape" response s only here
            if resp_in_set==-1; escape_response(OriginalCLUT); end 
            if resp_raw~=0
                % response, trial number, timestamp in expt, raw timestamp
                p.resp_raw_all = [p.resp_raw_all; [resp_raw, tt, ts-p.start_exp_time, ts]];
            end
        end
    
        % updating my timers here
        time_last_evt = time_last_evt + p.delay_time_sec;
        time_from_start = time_from_start + p.delay_time_sec;
            
        %% Text cues appear
 
        % plot left name
        % doing some formatting so that longer names go over two lines
        % (actually just applies to musical instrument)
        name = split(p.left_names{tt}, ' ');
        if length(name)>1
            left_text = sprintf('(1)\n%s\n%s', name{1}, name{2});
        else
            left_text = sprintf('(1)\n%s', name{1});
        end
        DrawFormattedText(w, left_text, ...
            'center', ...  
            'center',  p.text_color, ...
            [], [], [], [], [],...
            p.text_rect_l);
    
        % plot right name
        name = split(p.right_names{tt}, ' ');
        if length(name)>1
            right_text = sprintf('(2)\n%s\n%s', name{1}, name{2});
        else
            right_text = sprintf('(2 )\n%s', name{1});
        end
        DrawFormattedText(w, right_text, ...
            'center', ...
            'center',  p.text_color, ...
            [], [], [], [], [],...
            p.text_rect_r);
    
         
%         Screen('FrameRect', w, p.white, p.text_rect_l)
%         Screen('FrameRect ', w, p.white, p.text_rect_r)
    
     
        Screen('DrawDots', w, [0,0], p.fix_size_pix, p.fix_color, p.center_pix, 0); 
        Screen('DrawingFinished', w);
        [~, onset] = Screen('Flip', w); 
        p.stim_flips(tt,3) = onset; % onset of cues

        % start checking responses as soon as text comes up
        keep_checking = 1;
        
        % Update timing - track duration since the last event ended
        time_in_this_evt = GetSecs - time_last_evt; 

        while (time_in_this_evt < p.cue_time_sec) % loop until duration passes
            time_in_this_evt = (GetSecs - time_last_evt);       
            [resp_raw, resp_in_set, ts] = check_for_resp(p.keys,p.escape);

            % check for "escape" response
            if resp_in_set==-1; escape_response(OriginalCLUT); end 

            % check for actual task response
            if keep_checking && resp_in_set

                %they responded to this stim with 1-2
                p.response(tt) = resp_in_set;
                % RT is measured from when cues come on
                p.resp_time_from_onset(tt)= ts - p.stim_flips(tt,3);
                % now we have one response - stop checking for further
                % responses.
                keep_checking=0;
            end

            if resp_raw~=0
                % response, trial number, timestamp in expt, raw timestamp
                p.resp_raw_all = [p.resp_raw_all; [resp_raw, tt, ts-p.start_exp_time, ts]];
            end

        end
    
        % updating my timers here
        time_last_evt = time_last_evt + p.cue_time_sec;
        time_from_start = time_from_start + p.cue_time_sec;
        
        %% ITI starts now
        % just blank screen, but keep checking responses

        Screen('DrawDots', w, [0,0], p.fix_size_pix, p.fix_color, p.center_pix, 0); 
        Screen('DrawingFinished', w);                     
        [~, onset] = Screen('Flip', w); 
        p.stim_flips(tt,4) = onset; % offset of cues

        % Update timing - track duration since the last event ended
        time_in_this_evt = GetSecs - time_last_evt; 

        while (time_in_this_evt < p.iti_sec(tt)) % loop until duration passes
            time_in_this_evt = (GetSecs - time_last_evt);       
            [resp_raw, resp_in_set, ts] = check_for_resp(p.keys,p.escape);

            % check for "escape" response
            if resp_in_set==-1; escape_response(OriginalCLUT); end 

            % check for actual task response
            if keep_checking && resp_in_set

                %they responded to this stim with 1-2
                p.response(tt) = resp_in_set;
                % RT is measured from when cues come on
                p.resp_time_from_onset(tt)= ts - p.stim_flips(tt,3);
                % now we have one response - stop checking for further
                % responses.
                keep_checking=0;
            end

            if resp_raw~=0
                % response, trial number, timestamp in expt, raw timestamp
                p.resp_raw_all = [p.resp_raw_all; [resp_raw, tt, ts-p.start_exp_time, ts]];
            end

        end
    
        % updating my timers here
        time_last_evt = time_last_evt + p.iti_sec(tt);
        time_from_start = time_from_start + p.iti_sec(tt);
        

        % if they still haven't responded - this is when we would mark
        % it as a missed response.
        if keep_checking 
             keep_checking=0;
             p.response(tt) = 0;
        end
       
    end 
    
    %% finish experiment 
       
    % get accuracy
    trialsdone = ~isnan(p.stim_flips(:,4));
    acc = mean(p.response(trialsdone)==p.correct_resp(trialsdone));
    p.accuracy = acc;
    
    % final fixation:
    Screen('DrawDots', w, [0,0], p.fix_size_pix, p.fix_color, p.center_pix, 0); 
    Screen('DrawingFinished', w);
    Screen('Flip', w);


    % Update timing - track duration since the last event ended
    time_in_this_evt = GetSecs - time_last_evt; 
    while (time_in_this_evt < p.end_fix_sec) % loop until duration passes
        time_in_this_evt = (GetSecs - time_last_evt);       
        [resp_raw, resp_in_set, ts] = check_for_resp(p.keys,p.escape);
        % check for "escape" responses only here
        if resp_in_set==-1; escape_response(OriginalCLUT); end 
        if resp_raw~=0
            % response, trial number, timestamp in expt, raw timestamp
            p.resp_raw_all = [p.resp_raw_all; [resp_raw, 0, ts-p.start_exp_time, ts]];
        end
    end

    % updating my timers here
    time_last_evt = time_last_evt + p.end_fix_sec;
    time_from_start = time_from_start + p.end_fix_sec;

    % how long did the whole experiment take to run?
    p.time_from_start = time_from_start;

    % difference between end time and start time
    % this should approx match the above value (time_from_start)
    p.end_exp_time = GetSecs; 
    p.total_exp_time = (p.end_exp_time-p.start_exp_time); 
    p.total_exp_time_mins = p.total_exp_time/60; 

    %% get accuracy
    
    % print some feedback to command window
    fprintf('\nCompleted block %d!\n', p.run_num);
    
    fprintf('Accuracy is %.2f percent\n', p.accuracy * 100);

    fprintf('Number of time out trials: %d/%d\n', sum(p.response==0), p.num_trials);
    
    % and draw on the screen for subject to see
    InstrText = ['Block finished!' '\n\n'...
                sprintf('Accuracy is %.2f percent', p.accuracy*100)];
                
    DrawFormattedText(w, InstrText, 'center', 'center', p.white);
    % put up a message to wait
    Screen('DrawingFinished', w);
    Screen('Flip', w);
         
    %----------------------------------------------------------------------
    %SAVE OUT THE DATA-----------------------------------------------------
    %----------------------------------------------------------------------

    if p.run_num>1
        % if this is not the first run, then we want to make sure the 
        % dat_all list from previous runs is loaded here
       load(p.fnsave_local, 'dat_all');
    end

    % This is where we are appending the existing run into our dat_all list
    % If this is run 1, then we're creating dat_all from scratch here.
    dat_all(p.run_num).p = p;
    
    % save the file to disk here
    save(p.fnsave_local,'dat_all');
    
    resp_in_set=0; 
    % wait for a space bar press or escape to exit
    while resp_in_set==0
        [resp_raw, resp_in_set, ts] = check_for_resp(p.space, p.escape);
        if resp_in_set==-1; escape_response(OriginalCLUT); end 
    end

    KbReleaseWait();
    
    
    
    %----------------------------------------------------------------------
    %WINDOW CLEANUP--------------------------------------------------------
    %----------------------------------------------------------------------
    %This closes all visible and invisible screens and puts the mouse cursor
    %back on the screen
    Screen('CloseAll');
    if exist('OriginalCLUT','var')
        if exist('ScreenNumber','var')
            Screen('LoadCLUT', ScreenNumber, OriginalCLUT);
        else
            Screen('LoadCLUT', 0, OriginalCLUT);
        end
    end
    clear screen
    ListenChar(1);
    ShowCursor;
    
catch err
    
    %If an error occurred in the "try" block, this code is executed
   
    if exist('OriginalCLUT','var') && ~isempty(OriginalCLUT)
        if exist('ScreenNumber','var')
            Screen('LoadCLUT', ScreenNumber, OriginalCLUT);
        else
            Screen('LoadCLUT', 0, OriginalCLUT);
        end
    end
    Screen('CloseAll');                
    ShowCursor;
    if IsWin
        ShowHideWinTaskbarMex;     
    end
    ListenChar(1)
    rethrow(err)

end
