#=============================================================================
                             MISC
=============================================================================#

"""
    loadargs(fn::String)

Load arguments (.yml) YAML file.
"""
loadargs(fn::String) = YAML.load_file(fn; dicttype=Dict{Symbol,Any})

"""
    saveargs(fn::String, args::Dict{Symbol,Any})

Save arguments dictionary to (.yml) YAML file.
"""
saveargs(fn::String, args::Dict{Symbol,Any}) = YAML.write_file(fn, args)

"""
    setrecursive!(d::Dict, key, value)

In a (possibly nested) dictionary d, set the value of key to value. The search
for key is done greedily, so only setting of unique keys is recommended. An
error flag=true is returned if the key was not found in d.
"""
function setrecursive!(d::Dict, key, value)
	if haskey(d, key)
		d[key] = value
		return false
	end
	flag = true # key has not been set to value
	for k ∈ keys(d)
		if d[k] isa Dict
			flag = setrecursive!(d[k], key, value)
			!flag && break
		end
	end
	return flag
end

