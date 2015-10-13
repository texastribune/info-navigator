===
infoNavigator
===

infoNavigator takes your models and makes tables for all of them so that internal users can explore them

Quick start
-----------
1. Add "infoNavigator" to your INSTALLED_APPS setting like this::
    INSTALLED_APPS = (
        ...
        'polls',
    )

2. Get a url to all those models within your django project like this:
    from infoNavigator.views import RecordsView
    url(r'^navigate/', RecordsView(models.Keyword, models.SICCode, models.NAICSCode,
                                   models.OSHA170Form, models.ReportingJurisdiction,
                                   models.Accident, models.AccidentAbstract,
                                   models.AccidentInjury, models.Inspection,
                                   models.OptionalInspectionInfo, models.RelatedActivity,
                                   models.StrategicCode, models.Violation,
                                   models.ViolationEvent,
                                   models.ViolationGeneralDuty).as_view()),

3. Visit the url you set up