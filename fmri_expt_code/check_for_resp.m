function [resp_raw, resp_in_set, ts] = check_for_resp(resp_set, escape_key)

% queries the keyboard to see if a legit response was made
% returns the response and the timestamp
% this is a newer version that returns keypresses even if they're not one
% of the specified responses, which helps w debugging code.
% MMH oct 2023

[~, secs, keyCode] = KbCheck;

key_index = find(keyCode);

if numel(key_index)>=1   

    ts = secs;

    if numel(key_index)>1
        % in the case of multiple keypresses, just consider the first one
        key_index = key_index(1);
    end

    resp_raw = key_index;

    if ismember(key_index, resp_set)
        % returns the index into resp_set for this response
        resp_in_set = find(key_index==resp_set);
    elseif key_index==escape_key
        % escape key returns a negative 1
        resp_in_set = -1;
    else
        resp_in_set = 0;
    end

else

    resp_raw = 0;
    resp_in_set = 0;
    ts = nan;

end
