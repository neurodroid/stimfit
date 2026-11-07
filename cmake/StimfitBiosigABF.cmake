include_guard(GLOBAL)

# BioSig 4.0.0 can overrun its ABF2 string table when directory and SSCH counts
# differ. Apply the same bounded parsing fix to portable and staged MSVC sources,
# keeping the release submodule pristine. Remove this when upstream fixes it.
function(stf_patch_biosig_abf source_file output_file)
  file(READ "${source_file}" _source)
  set(_before [==[readABF2block(hdr->AS.Header + offsetof(struct ABF_FileInfo, StringsSection), hdr, &S);]==])
  set(_after [==[size_t stringSectionBytes = readABF2block(hdr->AS.Header + offsetof(struct ABF_FileInfo, StringsSection), hdr, &S);]==])
  string(FIND "${_source}" "${_before}" _position)
  if(_position LESS 0)
    message(FATAL_ERROR "Biosig ABF compatibility patch no longer matches ${source_file}")
  endif()
  string(REPLACE "${_before}" "${_after}" _source "${_source}")
  set(_before [==[		// build String Section CacHe (SSCH)
		char **ssch = calloc(S.llNumEntries, sizeof(char*) );
		for (size_t k=44, nstr=0; (nstr<S.llNumEntries) && (k<S.llNumEntries*S.uBytes);) {
			while (!StringSectionBlock[k]) k++;
			char *str=StringSectionBlock+k;
			size_t len=strlen(str);
			ssch[nstr] = str;
			nstr++;
			k+=len;
			if (VERBOSE_LEVEL>5) fprintf(stdout,"%d: (L%d)\t%s\n",nstr,len,str);
		}
]==])
  set(_after [==[        // Size the table from the SSCH header, independently of an unvalidated
        // directory count. Inconsistent counts must not cause an out-of-bounds read.
        size_t nstrings = 0;
        const char **ssch = NULL;
        if (stringSectionBytes < 44 || memcmp(StringSectionBlock, "SSCH", 4)) {
            free((void*)StringSectionBlock);
            biosigERROR(hdr, B4C_INCOMPLETE_FILE, "Invalid ABF2 string-cache header");
            return;
        }
        nstrings = leu32p(StringSectionBlock + 8);
        if (nstrings > stringSectionBytes - 44) {
            free((void*)StringSectionBlock);
            biosigERROR(hdr, B4C_INCOMPLETE_FILE, "Invalid ABF2 string count");
            return;
        }
        ssch = (const char**)calloc(nstrings, sizeof(char*));
        if (nstrings && !ssch) {
            free((void*)StringSectionBlock);
            biosigERROR(hdr, B4C_MEMORY_ALLOCATION_FAILED, "ABF2 string cache");
            return;
        }
        size_t stringOffset = 44;
        for (size_t nstr = 0; nstr < nstrings; ++nstr) {
            const char *str = StringSectionBlock + stringOffset;
            const char *end = (const char*)memchr(str, 0, stringSectionBytes - stringOffset);
            if (!end) {
                free(ssch);
                free((void*)StringSectionBlock);
                biosigERROR(hdr, B4C_INCOMPLETE_FILE, "Unterminated ABF2 string");
                return;
            }
            // Empty strings occupy an index too.
            ssch[nstr] = str;
            stringOffset = (size_t)(end - StringSectionBlock) + 1;
        }]==])
  string(FIND "${_source}" "${_before}" _position)
  if(_position LESS 0)
    message(FATAL_ERROR "Biosig ABF compatibility patch no longer matches ${source_file}")
  endif()
  string(REPLACE "${_before}" "${_after}" _source "${_source}")
  set(_before [==[strncpy(hc->Label, ssch[ix-1], MAX_LENGTH_LABEL+1);]==])
  set(_after [==[strncpy(hc->Label, (ix > 0 && (size_t)ix <= nstrings) ? ssch[ix-1] : "", MAX_LENGTH_LABEL+1);]==])
  string(FIND "${_source}" "${_before}" _position)
  if(_position LESS 0)
    message(FATAL_ERROR "Biosig ABF compatibility patch no longer matches ${source_file}")
  endif()
  string(REPLACE "${_before}" "${_after}" _source "${_source}")
  set(_before [==[hc->PhysDimCode = PhysDimCode(ssch[ix-1]);]==])
  set(_after [==[hc->PhysDimCode = PhysDimCode((ix > 0 && (size_t)ix <= nstrings) ? ssch[ix-1] : "");]==])
  string(FIND "${_source}" "${_before}" _position)
  if(_position LESS 0)
    message(FATAL_ERROR "Biosig ABF compatibility patch no longer matches ${source_file}")
  endif()
  string(REPLACE "${_before}" "${_after}" _source "${_source}")
  if(EXISTS "${output_file}")
    file(READ "${output_file}" _existing)
  else()
    set(_existing "")
  endif()
  if(NOT _existing STREQUAL _source)
    file(WRITE "${output_file}" "${_source}")
  endif()
endfunction()
