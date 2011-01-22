#
# Regular cron jobs for the stimfit package
#
0 4	* * *	root	[ -x /usr/bin/stimfit_maintenance ] && /usr/bin/stimfit_maintenance
